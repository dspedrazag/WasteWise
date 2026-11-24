"""Standalone WasteWise prototype, extracted from 06_wastewise_app_prototype.ipynb.

Run from application/ with the project's scientific Python interpreter:
    python app.py

Uses the existing checkpoint and workbook in the parent project. Uploads are
session-only temporary files; no prediction history or image logging is enabled.
"""

import os
import sys
import json
import io
import warnings
import re
import importlib.metadata as metadata
import tempfile
import signal
from threading import Lock
from pathlib import Path
from copy import deepcopy
from functools import partial
from urllib.parse import urlsplit
from html import escape

# Disable optional telemetry before importing Gradio. Do not write local .pyc files.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["GRADIO_PROFILING"] = "0"
# Gradio must see this before it is imported. Only this process owns this cache.
# Temporary uploads are removed on session disconnect and normal process exit.
UPLOAD_CACHE = tempfile.TemporaryDirectory(prefix="wastewise-uploads-")
UPLOAD_ROOT = Path(UPLOAD_CACHE.name).resolve()
os.environ["GRADIO_TEMP_DIR"] = str(UPLOAD_ROOT)
sys.dont_write_bytecode = True

required_packages = ["torch", "torchvision", "pandas", "Pillow", "gradio", "openpyxl", "prometheus-client"]
missing = []
versions = {}
for package in required_packages:
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        missing.append(package)
if missing:
    raise ImportError("Missing application packages: " + ", ".join(missing)
                      + ". Install them in the scientific environment after approval, then restart the application.")

import pandas as pd
import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.transforms import InterpolationMode
from PIL import Image, UnidentifiedImageError
import gradio as gr
from prometheus_client import CollectorRegistry, Counter, Histogram, start_http_server

# application/ sits directly below the existing project root; cwd is irrelevant.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import file_sha256
from src.experiment_training import restore_checkpoint

RUN_DIRECTORY = PROJECT_ROOT / "results/experiments/efficientnet_b0/baseline_seed42_5epochs"
RULES_PATH = PROJECT_ROOT / "data/WasteWise_UF_Recycling_Rules.xlsx"
EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
# User-provided pilot information; not independently verified official inventories.
PILOT_BINS = {
    "UF NPB": ("Bottles & Cans", "Paper & Cardboard", "Waste / Landfill"),
    "UF MALA": ("Bottles & Cans", "Waste / Landfill"),
}
PILOT_LOCATIONS = list(PILOT_BINS)
PILOT_INVENTORY_NOTE = "User-provided pilot information; not independently verified."
MIN_CONFIDENCE = 0.85  # Provisional development threshold; Softmax is uncalibrated.
assert 0 < MIN_CONFIDENCE <= 1
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
DEVICE = torch.device("cpu")
HOST = "127.0.0.1"
PORT = None  # Gradio finds a free local port, as in the notebook.

# Aggregate numbers only: no image, filename, session, location or user labels.
METRICS_REGISTRY = CollectorRegistry()
INFERENCE_LATENCY = Histogram(
    "wastewise_inference_duration_seconds",
    "CPU model call and softmax duration, excluding image decoding, preprocessing and queue time.",
    registry=METRICS_REGISTRY,
)
PREDICTION_COUNT = Counter(
    "wastewise_predictions_total",
    "Completed material predictions, including below-threshold predictions; cached reuse is excluded.",
    labelnames=["material"],
    registry=METRICS_REGISTRY,
)
for material in EXPECTED_CLASSES:
    PREDICTION_COUNT.labels(material=material)  # Expose zero counts for all six classes.
WITHHOLDING_COUNT = Counter(
    "wastewise_withholding_total",
    "Withholding events: low confidence, rejected material, or a withheld/out-of-scope rule decision.",
    registry=METRICS_REGISTRY,
)
DECISION_COUNT = Counter(
    "wastewise_decisions_total",
    "First completed outcome per scan attempt; unfinished workflows and processing errors are excluded.",
    labelnames=["outcome"],
    registry=METRICS_REGISTRY,
)
for outcome in ("withheld", "guidance"):
    DECISION_COUNT.labels(outcome=outcome)



def load_trained_model(directory=RUN_DIRECTORY):
    directory = Path(directory)
    for filename in ["best_model.pt", "config.json", "metrics.json", "complete.json"]:
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"Required trained-model file is missing: {directory / filename}")
    complete = json.loads((directory / "complete.json").read_text())
    if complete.get("status") != "complete":
        raise ValueError("The selected model run is not marked complete.")
    for filename in ["best_model.pt", "config.json", "metrics.json"]:
        if file_sha256(directory / filename) != complete["sha256"].get(filename):
            raise ValueError(f"Saved model artifact failed its checksum: {filename}")
    config = json.loads((directory / "config.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    if (config["model_name"] != "efficientnet_b0" or config["loss"] != "CrossEntropyLoss"
            or config["run_name"] != "baseline_seed42_5epochs" or config["classes"] != EXPECTED_CLASSES):
        raise ValueError("Model identity, unweighted run, or class ordering does not match the selected experiment.")

    model = efficientnet_b0(weights=None)
    model.classifier = nn.Linear(model.classifier[-1].in_features, len(config["classes"]))
    run = {"directory": directory, "config": config, "metrics": metrics}
    identity = {"classes": config["classes"], "manifest_sha256": config["manifest_sha256"]}
    model = restore_checkpoint(model, run, identity, DEVICE)
    model.requires_grad_(False)

    saved = config["preprocessing"]
    transform = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms(
        resize_size=saved["resize_size"][0], crop_size=saved["crop_size"][0],
        interpolation=InterpolationMode(saved["interpolation"]), antialias=saved["antialias"],
        mean=saved["mean"], std=saved["std"],
    )
    return model, transform, list(config["classes"]), config

model, preprocess, classes, model_config = load_trained_model()


def read_image(image_input):
    # With preprocessing disabled on the photo event, validate the original upload.
    if isinstance(image_input, dict):
        image_input = image_input.get("path")
    if image_input is None:
        raise ValueError("Upload or take a photo before continuing.")
    if isinstance(image_input, (bytes, bytearray)):
        data = bytes(image_input)
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("The image exceeds the configured 5 MB upload limit.")
        source = io.BytesIO(data)
    else:
        path = Path(image_input)
        if not path.is_file():
            raise ValueError("The uploaded image is unavailable. Please upload it again.")
        if path.stat().st_size > MAX_UPLOAD_BYTES:
            raise ValueError("The image exceeds the configured 5 MB upload limit.")
        source = path
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as image:
                if image.format not in SUPPORTED_FORMATS:
                    raise ValueError("Use a JPEG, PNG, or WebP image.")
                if image.width * image.height > 20_000_000:
                    raise ValueError("The image is too large to process. Use a smaller photo.")
                image.load()  # Decode now so corrupted images fail before prediction.
                return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("This image could not be read. Try a valid JPEG, PNG, or WebP photo.") from error


def predict_material(image_input):
    image = read_image(image_input)
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with INFERENCE_LATENCY.time(), torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].cpu()
    if probabilities.shape != (len(classes),) or not torch.isfinite(probabilities).all():
        raise ValueError("The model did not produce a valid material prediction.")
    index = int(probabilities.argmax())
    PREDICTION_COUNT.labels(material=classes[index]).inc()
    return {"material": classes[index], "confidence": float(probabilities[index]),
            "probabilities": dict(zip(classes, probabilities.tolist()))}


REQUIRED_COLUMNS = {
    "UF Rules": ["rule_id", "item_type", "destination", "uf_acceptance", "preparation_guidance",
                 "source_url", "source_ids", "review_date", "review_notes", "ai_material_candidate", "interaction_scope",
                 "general_bin_category", "component_instructions"],
    "Sources": ["Source ID", "UF official page", "Source URL", "Reviewed on"],
    "Item Options": ["option_id", "AI material", "user-facing option (English)", "source_rule_id",
                     "MVP interface scope", "selecting this option", "reviewed_on"],
    "Condition Paths": ["source_rule_id", "AI material", "requires state question", "condition question",
                        "YES outcome", "YES message", "NO outcome", "NO message",
                        "NOT SURE outcome", "NOT SURE message", "MVP scope gate"],
    "Cardboard Steps": ["rule_id", "step", "question", "answer", "next_step", "outcome", "display_instruction"],
}


def load_rules(path=RULES_PATH):
    if not Path(path).is_file():
        raise FileNotFoundError(f"UF recycling rules workbook is missing: {path}")
    tables = pd.read_excel(path, sheet_name=list(REQUIRED_COLUMNS), engine="openpyxl", dtype=str)
    for name, columns in REQUIRED_COLUMNS.items():
        tables[name] = tables[name].fillna("")
        absent = set(columns) - set(tables[name].columns)
        if absent:
            raise ValueError(f"{name}: missing columns {sorted(absent)}")
        keys = ["rule_id", "step", "answer"] if name == "Cardboard Steps" else [columns[0]]
        if tables[name].duplicated(keys).any() or tables[name][keys].eq("").any().any():
            raise ValueError(f"{name}: missing or duplicate rule keys; guidance is disabled.")
    rule_ids = set(tables["UF Rules"]["rule_id"])
    referenced = set(tables["Item Options"]["source_rule_id"]) - {""}
    if not referenced <= rule_ids:
        raise ValueError("Item Options refers to missing UF rules.")
    if not set(tables["Cardboard Steps"]["rule_id"]) <= rule_ids:
        raise ValueError("Cardboard Steps refers to missing UF rules.")
    rules = tables["UF Rules"].set_index("rule_id").to_dict("index")
    paths = tables["Condition Paths"].set_index("source_rule_id").to_dict("index")
    # A Not accepted rule cannot become recyclable through a condition branch.
    for rule_id, rule in rules.items():
        if rule["uf_acceptance"] == "Not accepted":
            outcomes = [paths.get(rule_id, {}).get(f"{answer} outcome", "")
                        for answer in ("YES", "NO", "NOT SURE")]
            outcomes += tables["Cardboard Steps"].loc[
                tables["Cardboard Steps"]["rule_id"] == rule_id, "outcome"].tolist()
            if "CONFIRM_RECYCLE" in outcomes:
                raise ValueError(f"Not accepted rule {rule_id} has a CONFIRM_RECYCLE condition branch; guidance is disabled.")
    for option in tables["Item Options"].to_dict("records"):
        rule_id = option["source_rule_id"] or option["option_id"]
        if rule_id not in paths:
            raise ValueError(f"Missing condition path for {option['option_id']}")
        if (paths[rule_id]["AI material"] != option["AI material"]
                or paths[rule_id]["MVP scope gate"] != option["MVP interface scope"]):
            raise ValueError(f"Conflicting material/scope mapping for {rule_id}")
        if option["source_rule_id"] and (
            rules[rule_id]["ai_material_candidate"] != option["AI material"]
            or rules[rule_id]["interaction_scope"] != option["MVP interface scope"]
        ):
            raise ValueError(f"UF rule disagrees with the option mapping for {rule_id}")
    sources = tables["Sources"].set_index("Source ID").to_dict("index")
    for rule in rules.values():
        ids = [key.strip() for key in rule["source_ids"].split(";") if key.strip()]
        if not ids or any(key not in sources for key in ids):
            raise ValueError("A UF rule refers to a missing source.")
        if {sources[key]["Source URL"] for key in ids} != {url.strip() for url in rule["source_url"].split("|")}:
            raise ValueError("UF rule source IDs and URLs disagree.")
    return tables


tables = load_rules()
official_sources = tables["Sources"].set_index("Source ID").to_dict("index")
# Read the approved fallback documentation without changing any decision branch.
workbook_notes = pd.read_excel(RULES_PATH, sheet_name="Start Here", usecols=[0, 1], dtype=str).fillna("")
notes_by_label = dict(workbook_notes.itertuples(index=False, name=None))
general_fallback = notes_by_label.get("Other / Not sure: general fallback", "")
if not general_fallback or "S5" not in official_sources:
    raise ValueError("The workbook's documented conservative fallback or FAQ source is missing.")
uf_rules = tables["UF Rules"].set_index("rule_id").to_dict("index")
item_options = tables["Item Options"].set_index("option_id").to_dict("index")
condition_paths = tables["Condition Paths"].set_index("source_rule_id").to_dict("index")
cardboard_steps = tables["Cardboard Steps"].to_dict("records")


OTHER = "__other__"
ACTIVE_SCOPE = "MVP - proposed active"
NEGATIVE_SCOPE = "MVP - negative/safety option"
INFORMATION_OUTCOMES = {"CONFIRM_RECYCLE", "PREPARE_FIRST", "NOT_ACCEPTED_AS_IS", "NOT_ACCEPTED_IN_STREAM"}
# Existing ordinary-waste items without a recorded trash destination. Default deny:
# new items, special waste and withheld outcomes never inherit this display fallback.
ORDINARY_WASTE_FALLBACK_RULES = {
    "corrugated_cardboard_box", "cracker_cereal_box", "empty_pizza_box",
    "greasy_cardboard", "food_soiled_paper",
}


def item_choices(material):
    matching = [(key, row) for key, row in item_options.items()
                if row["AI material"].lower() == material and row["source_rule_id"]]
    priority = {ACTIVE_SCOPE: 0, NEGATIVE_SCOPE: 1}
    matching.sort(key=lambda pair: priority.get(pair[1]["MVP interface scope"], 2))
    choices = []
    for key, row in matching:
        label = row["user-facing option (English)"]
        if row["selecting this option"] == "WITHHOLD_CONFLICT":
            label += " — unresolved UF rule"
        elif row["MVP interface scope"] not in (ACTIVE_SCOPE, NEGATIVE_SCOPE):
            label += " — not enabled in this prototype"
        choices.append((label, key))
    return choices + [("Other / Not sure", OTHER)]


def questions_for(option_id):
    option = item_options.get(option_id)
    if not option or not option["source_rule_id"]:
        return []
    rule_id = option["source_rule_id"]
    rows = [row for row in cardboard_steps if row["rule_id"] == rule_id]
    if rows:
        questions = []
        # Preserve the worksheet's documented order; answers/transitions also come from its rows.
        for step in dict.fromkeys(row["step"] for row in rows):
            group = [row for row in rows if row["step"] == step]
            if len({row["question"] for row in group}) != 1:
                raise ValueError("Conflicting question wording in Cardboard Steps.")
            questions.append({"key": step, "text": group[0]["question"],
                              "choices": [row["answer"] for row in group]})
        return questions
    path = condition_paths[rule_id]
    if path["requires state question"] == "Yes":
        return [{"key": "condition", "text": path["condition question"],
                 "choices": ["Yes", "No", "Not sure"]}]
    return []


def rule_decision(material, option_id, answers, location, material_uncertain=False):
    def result(outcome, message, rule_id=None, question=None):
        return {"outcome": outcome, "message": message, "rule_id": rule_id, "question": question}

    if location not in PILOT_LOCATIONS:
        return result("WITHHOLD", "Select one of the two pilot labels to continue.")
    if material_uncertain or material not in classes:
        return result("WITHHOLD", "The material is unresolved. Try another clear photo.")
    option = item_options.get(option_id)
    if option_id == OTHER or (option and not option["source_rule_id"]):
        return result("WITHHOLD_UNKNOWN_ITEM", "The item is unresolved. Other / Not sure does not establish a disposal rule.")
    if not option or option["AI material"].lower() != material:
        return result("WITHHOLD", "Choose an item compatible with the predicted material. Material alone does not determine disposal.")
    rule_id = option["source_rule_id"]
    rule = uf_rules.get(rule_id)
    path = condition_paths.get(rule_id)
    if not rule or not path:
        return result("WITHHOLD", "The selected item has no complete UF rule and condition path.")
    if option["selecting this option"] == "WITHHOLD_CONFLICT" or rule["uf_acceptance"] == "Needs clarification":
        return result("WITHHOLD_CONFLICT", "The workbook records conflicting UF guidance for this item. No recommendation is available.", rule_id)
    if option["MVP interface scope"] not in (ACTIVE_SCOPE, NEGATIVE_SCOPE):
        return result("OUT_OF_SCOPE_MVP", "This item has reference information but is not enabled for guidance in this prototype.", rule_id)

    questions = questions_for(option_id)
    rows = [row for row in cardboard_steps if row["rule_id"] == rule_id]
    if rows:
        by_step = {q["key"]: q for q in questions}
        step = questions[0]["key"]
        for _ in questions:
            question = by_step[step]
            answer = answers.get(step)
            if not answer:
                return result("ASK_CONDITION", "Answer the required question.", rule_id, question)
            matches = [r for r in rows if r["step"] == step and r["answer"] == answer]
            if len(matches) != 1:
                return result("WITHHOLD_CONFLICT", "Missing or conflicting condition branch.", rule_id)
            branch = matches[0]
            if branch["outcome"] != "ASK_CONDITION":
                return result(branch["outcome"], branch["display_instruction"], rule_id)
            step = branch["next_step"]
            if step not in by_step:
                break
        return result("WITHHOLD_CONFLICT", "The workbook question sequence is incomplete or cyclic.", rule_id)
    if questions:
        question = questions[0]
        answer = answers.get(question["key"])
        if not answer:
            return result("ASK_CONDITION", "Answer the required question.", rule_id, question)
        if answer not in question["choices"]:
            return result("WITHHOLD", "The required answer is not recognized.", rule_id)
        column = answer.upper()
        return result(path[f"{column} outcome"], path[f"{column} message"], rule_id)
    if option["selecting this option"] == "NOT_ACCEPTED_IN_STREAM":
        return result("NOT_ACCEPTED_IN_STREAM", path["YES message"], rule_id)
    return result("WITHHOLD", "No supported interaction path was found.", rule_id)


def instruction_steps(text):
    """Split workbook prose into short steps without inventing actions."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part.strip()]


def location_note(location):
    bins = PILOT_BINS.get(location)
    description = "Available bins: " + ", ".join(bins) if bins else "Select a location to see its available bins."
    return description + "\n\n" + PILOT_INVENTORY_NOTE


def pilot_bin_name(category):
    """Match existing UF destination labels to the reported pilot bin labels."""
    if category.startswith("Trash / Landfill"):
        return "Waste / Landfill"
    return {"Bottles & Cans": "Bottles & Cans", "Paper Only": "Paper & Cardboard",
            "Cardboard container": "Paper & Cardboard", "Paper/Cardboard container": "Paper & Cardboard"}.get(category, "")


def component_html(text, location=None, option_id=None):
    """Show recorded actions; unresolved components never become disposal instructions."""
    if not text:
        return ""
    parts = re.split(r"(?:^|(?<=\.)\s+)([A-Z][A-Za-z /-]{1,35}):\s*", text)
    components = list(zip(parts[1::2], parts[2::2])) if len(parts) > 1 else [("Separated contents", text)]
    cards, unresolved = [], []
    for name, body in components:
        # Conservatively suppress the entire component when its evidence is unresolved.
        if re.search(r"unresolved|unverified|not implemented|mandatory removal is not established|do not present", body, re.I) or name == "Unresolved":
            unresolved.append(name)
            continue
        sentences = instruction_steps(body)
        known = [s for s in sentences if not re.search(r"destinations? (?:are |is )?not established", s, re.I)]
        if not known:
            unresolved.append(name)
            continue
        badge = '<span class="ww-evidence">Interpretation</span>' if "interpretation" in body.lower() else ""
        unknown_destination = '<p class="ww-muted">Destination not established for the separated contents.</p>' if len(known) != len(sentences) else ""
        bin_note = ""
        # The existing cup rules permit conditional sleeve recycling, independently of the cup.
        if "Paper/Cardboard" in body and location in PILOT_BINS:
            if "Paper & Cardboard" in PILOT_BINS[location]:
                bin_note = "Paper & Cardboard is listed in the user-provided pilot inventory for this location."
            else:
                bin_note = (f"The required Paper & Cardboard bin is unavailable at {location} according to the pilot inventory. "
                            "For a sleeve that meets these conditions, keep it temporarily until you can find an appropriate "
                            "Paper & Cardboard recycling bin elsewhere at UF. "
                            "Do not put the sleeve in Waste / Landfill solely because this bin is unavailable. "
                            "No alternative collection location is established by this information.")
        component_text = " ".join(known).replace("Trash / Landfill", "Waste / Landfill")
        cards.append(f'<div class="ww-component"><h4>{escape(name)} {badge}</h4>'
                     f'<p>{escape(component_text)}</p>{unknown_destination}'
                     + (f'<p class="ww-muted">{escape(bin_note)}</p>' if bin_note else "") + '</div>')
    result = ""
    if cards:
        scope = ("For ordinary disposable cups only; these component instructions do not apply to plates. "
                 if option_id == "paper_cup_plate" else "")
        result = ('<section class="ww-components"><h3>Separate the components</h3>'
                  '<p class="ww-muted">' + escape(scope) + 'If present and applicable to your selected item. '
                  'These components were not detected by AI.</p>' + "".join(cards) + '</section>')
    if unresolved:
        result += ('<p class="ww-component-hold"><strong>Component guidance not confirmed:</strong> '
                   + escape(", ".join(unresolved)) + '. No disposal or removal instruction is provided.</p>')
    return result


def result_text(state):
    """Render the existing decision as HTML. This function never changes rule outcomes."""
    decision = state.get("result") or {}
    valid = prediction_confirmed(state) and state["location"] in PILOT_LOCATIONS
    if valid:
        current = rule_decision(state["prediction"]["material"], state["item_id"], state["answers"],
                                state["location"], state["material_uncertain"])
        valid = all(decision.get(k) == current.get(k) for k in ("outcome", "rule_id"))
    if not valid:
        decision = {"outcome": "WITHHOLD", "rule_id": None,
                    "message": "Review the current photo and item selection before continuing."}
    outcome = decision.get("outcome", "WITHHOLD")
    rule = uf_rules.get(decision.get("rule_id"), {})
    option = item_options.get(state.get("item_id"), {})
    category = rule.get("general_bin_category", "").strip()
    tone, icon, heading = {
        "CONFIRM_RECYCLE": ("recycle", "♻", "Recycle"),
        "PREPARE_FIRST": ("caution", "⚠", "Prepare before recycling"),
        "NOT_ACCEPTED_AS_IS": ("exclude", "✕", "Do not recycle"),
        "NOT_ACCEPTED_IN_STREAM": ("exclude", "✕", "Do not recycle"),
        "WITHHOLD_SAFETY": ("caution", "⚠", "Special disposal may be needed"),
    }.get(outcome, ("caution", "⚠", "Recycling could not be confirmed"))
    destination, destination_label = "", "General bin category"
    if outcome == "CONFIRM_RECYCLE":
        destination = category
        summary = "Your item and answers meet the recorded UF recycling conditions."
        instructions = rule.get("preparation_guidance", "")
        # Omit only recorded actions already covered by this accepted item's answers.
        completed = {
            "plastic_beverage_bottle": ["Empty remaining beverage."],
            "aluminum_beverage_can": ["Empty remaining contents."],
            "plastic_personal_care_bottle": ["Ensure the bottle is empty and safe.",
                                           "Do not empty unknown or hazardous chemicals."],
            "plastic_cleaner_bottle": ["Ensure the bottle is empty and safe.",
                                     "Do not empty unknown or hazardous chemicals."],
            "corrugated_cardboard_box": ["Remove remaining contents."],
            "cracker_cereal_box": ["Remove remaining contents and separate non-cardboard contents.",
                                  "Recycle only when existing contamination and coating conditions pass."],
            "empty_pizza_box": ["Remove food remains.",
                               "Check for food residue, grease, oil, wax and plastic coatings.",
                               "Do not recycle when existing contamination conditions fail."],
        }
        for sentence in completed.get(decision.get("rule_id"), []):
            instructions = instructions.replace(sentence, "")
        if decision.get("rule_id") == "corrugated_cardboard_box":
            # Dryness is recorded guidance but is not established by either question.
            instructions = instructions.replace(
                "Recycle only if clean, dry, and free of food residue, grease, oil, wax and plastic coatings.",
                "Recycle only when dry.")
        elif decision.get("rule_id") == "cracker_cereal_box":
            instructions = instructions.replace("Flatten the clean cardboard.", "Flatten the cardboard.")
        elif decision.get("rule_id") == "empty_pizza_box":
            instructions = instructions.replace("Flatten only if eligible for recycling.", "Flatten the box.")
    elif outcome == "PREPARE_FIRST":
        # Do not expose a bin while later workbook questions are still unanswered.
        if all(q["key"] in state["answers"] for q in questions_for(state["item_id"])):
            destination = category
            destination_label = "Conditional destination — after all required conditions are satisfied"
        summary = "This item is not ready for recycling. Prepare it, then review the required answers."
        instructions = (rule.get("preparation_guidance", "") or decision["message"])
        if any(row["rule_id"] == decision.get("rule_id") for row in cardboard_steps):
            # Q1=No stops before cleanliness is known: empty first, then review the questions.
            instructions = instruction_steps(instructions)[0]
        instructions = "Do not recycle yet.\n" + instructions
        instructions += "\nReview your answers after preparation. All required conditions must be satisfied before recycling."
    elif outcome == "NOT_ACCEPTED_IN_STREAM":
        summary = "This item must stay out of the selected UF recycling stream."
        # A recycling destination must never leak into a negative result.
        if category.startswith("Trash / Landfill"):
            destination = category
            instructions = rule.get("preparation_guidance", "")
            if decision.get("rule_id") == "paper_cup_plate":
                instructions = ("For a plate, this rule does not establish an alternative disposal destination; "
                                "consult the official UF guidance below.\n" + instructions)
            instructions = "The recorded UF rules exclude this selected item from recycling.\n" + instructions
        else:
            instructions = decision["message"]
    elif outcome == "NOT_ACCEPTED_AS_IS":
        summary = "The reported condition does not meet the UF recycling requirements."
        # Preserve the exclusion. The ordinary-waste fallback below is separate
        # from the workbook branch and does not change its outcome.
        instructions = "\n".join(s for s in instruction_steps(decision["message"])
                                 if not re.search(r"Do not (?:infer|assign) a trash destination", s))
        if category.startswith("Trash / Landfill"):
            destination = category
    else:
        summary = decision.get("message", "No applicable UF rule is available.")
        if outcome == "WITHHOLD_SAFETY":
            summary = summary.replace("Do not recommend recycling", "Recycling cannot be recommended")
            summary = summary.replace("do not suggest pouring chemicals out.",
                                      "Do not pour out unknown or hazardous chemicals.")
        instructions = ("Consult the official UF guidance below before disposing of this item."
                        if outcome in {"WITHHOLD_SAFETY", "WITHHOLD_CONFLICT", "OUT_OF_SCOPE_MVP"}
                        else "Review your item selection and answers. If the material looks wrong, try another clear photo. "
                             "If the item or its condition remains uncertain, consult the official UF guidance before disposal.")
    # Apply only to current, eligible negative outcomes for the reviewed ordinary items.
    # This is the user-approved presentation fallback, not a new item-specific UF rule.
    ordinary_fallback = (valid and outcome in {"NOT_ACCEPTED_AS_IS", "NOT_ACCEPTED_IN_STREAM"}
                         and decision.get("rule_id") in ORDINARY_WASTE_FALLBACK_RULES
                         and not destination)
    if ordinary_fallback:
        destination = "Trash / Landfill"
        instructions += "\nUse Trash / Landfill for ordinary non-hazardous waste when no separate collection route applies."
    if outcome == "WITHHOLD_UNKNOWN_ITEM":
        summary = "WasteWise does not have enough information or an applicable rule to confirm this item's recyclability."
    # Availability changes the local recommendation, never the workbook outcome or safety gates.
    availability_note = ""
    if destination:
        required_bin = pilot_bin_name(destination)
        if required_bin and required_bin in PILOT_BINS.get(state["location"], ()):
            qualifier = destination[len("Trash / Landfill"):] if destination.startswith("Trash / Landfill") else ""
            destination = required_bin + qualifier
            if outcome != "PREPARE_FIRST":
                destination_label = "Available pilot bin"
        else:
            if required_bin:
                availability_note = (f"The required {required_bin} bin is unavailable at {state['location']} "
                                     "according to the user-provided pilot inventory.")
            else:
                availability_note = f"The pilot inventory does not establish a local bin for {destination}."
            availability_note += " No alternative collection location is established by this information."
            destination = ""
            if outcome == "CONFIRM_RECYCLE":
                tone, icon = "caution", "⚠"
                heading = "Recyclable — required bin unavailable" if required_bin else "Recyclable — local bin not established"
            if outcome in {"CONFIRM_RECYCLE", "PREPARE_FIRST"}:
                if required_bin:
                    instructions += (f"\nKeep the item temporarily until you can find an appropriate {required_bin} "
                                     "recycling bin elsewhere at UF.")
                else:
                    instructions += "\nConsult the official UF guidance to identify an appropriate recycling bin."
                instructions += "\nDo not use Waste / Landfill solely because the required recycling bin is unavailable."
    instructions = instructions.replace("Trash / Landfill", "Waste / Landfill")
    destination_html = (f'<div class="ww-destination"><span>{escape(destination_label)}</span>'
                        f'<strong>{escape(destination)}</strong></div>') if destination else ""
    if availability_note:
        destination_html += f'<p class="ww-destination-note">{escape(availability_note)}</p>'
    if ordinary_fallback:
        destination_html += ('<p class="ww-destination-note"><strong>Conservative ordinary-waste fallback.</strong> '
                             'General contamination-prevention guidance, not a verified item-specific disposal rule. '
                             'Excludes chemicals, batteries, electronics, laboratory, medical and other special waste; '
                             'seek official guidance for those items.</p>')
    banner = (f'<section class="ww-outcome ww-outcome-{tone}" aria-label="Disposal outcome">'
              f'<div class="ww-outcome-title"><span aria-hidden="true">{icon}</span><h3>{heading}</h3></div>'
              f'<p>{escape(summary)}</p>{destination_html}</section>')
    steps = list(dict.fromkeys(instruction_steps(instructions)))
    actions = '<h3>What should you do?</h3><ol>' + "".join(f'<li>{escape(s)}</li>' for s in steps) + '</ol>'
    if outcome in {"CONFIRM_RECYCLE", "NOT_ACCEPTED_IN_STREAM"}:
        actions += component_html(rule.get("component_instructions", ""), state["location"], state.get("item_id"))
    fallback = outcome == "WITHHOLD_UNKNOWN_ITEM" and valid
    if fallback:
        # Retain the S5 source and its limits without suggesting a bin for an unknown item.
        body = ("The UF FAQ's general guidance for ordinary non-hazardous waste is not a verified item-specific rule "
                "for your selection. This screen does not establish whether your item qualifies. "
                "That general guidance excludes chemicals, batteries, electronics, medical waste, laboratory waste "
                "and other special-disposal items.")
        actions += ('<section class="ww-fallback"><h3>No item-specific disposal recommendation</h3>'
                    f'<p>{escape(body)}</p>'
                    '<p>Consult the official UF guidance before deciding how to dispose of this item.</p></section>')
    material = (state.get("prediction") or {}).get("material", "Not available").title()
    item_label = option.get("user-facing option (English)", "Other / Not sure" if state.get("item_id") == OTHER else "Not selected")
    review = ('<aside class="ww-item-review"><h3>Your item review</h3><dl>'
              f'<dt>AI-predicted material</dt><dd>{escape(material)}</dd>'
              f'<dt>Selected item</dt><dd>{escape(item_label)}</dd></dl>')
    # Use actual workbook questions; Q1/Q2 are internal keys only.
    try:
        questions = questions_for(state.get("item_id")) if option else []
    except (KeyError, ValueError):
        questions = []
    answered = [q for q in questions if q["key"] in state["answers"]]
    if answered:
        review += '<dl class="ww-answers">' + "".join(
            f'<dt>{escape(q["text"])}</dt><dd>{escape(state["answers"][q["key"]])}</dd>' for q in answered) + '</dl>'
    review += '</aside>'
    source_ids = [s.strip() for s in rule.get("source_ids", "").split(";") if s.strip()]
    if fallback or ordinary_fallback:
        source_ids.append("S5")
    if not source_ids:
        source_ids.append("S3")
    links, source_reviews = [], []
    for source_id in dict.fromkeys(source_ids):
        source = official_sources.get(source_id, {})
        url = source.get("Source URL", "")
        parsed = urlsplit(url)
        if parsed.scheme == "https" and (parsed.hostname or "").endswith(".ufl.edu"):
            links.append(f'<li><a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">'
                         f'{escape(source["UF official page"])}</a>'
                         f'<span> · reviewed {escape(source["Reviewed on"])}</span></li>')
            source_reviews.append(source["UF official page"] + ": " + source["Reviewed on"])
    dates = []
    if rule.get("review_date"):
        dates.append("Rule reviewed " + rule["review_date"])
    if option.get("reviewed_on"):
        dates.append("Interaction reviewed " + option["reviewed_on"])
    guidance_dates = re.findall(r"Guidance review[: ]+(\d{4}-\d{2}-\d{2})",
                               rule.get("review_notes", "") + (" " + general_fallback if fallback or ordinary_fallback else ""))
    if guidance_dates:
        dates.append("Guidance reviewed " + max(guidance_dates))
    details = '<p>' + escape(" · ".join(dates)) + '</p>' if dates else ""
    if source_reviews:
        details += '<p><strong>Source review dates:</strong> ' + escape(" · ".join(source_reviews)) + '</p>'
    # Raw review notes may mention disabled destinations: show them only alongside
    # already-supported item guidance, never on a scope, conflict or safety hold.
    if outcome in {"CONFIRM_RECYCLE", "NOT_ACCEPTED_IN_STREAM"} and rule.get("review_notes"):
        details += f'<p>{escape(rule["review_notes"])}</p>'
    education = ('<section class="ww-education"><h3>Help prevent recycling contamination</h3>'
                 '<p>Food, liquids, and non-recyclable materials can contaminate recycling. '
                 'Follow the preparation steps and keep excluded materials out.</p>'
                 '<p class="ww-muted">WasteWise uses AI to identify the material. Disposal guidance depends on '
                 'your item selection, answers, and applicable UF recycling rules.</p></section>')
    return ('<article class="ww-result"><h2>Your disposal recommendation</h2>' + banner
            + f'<p class="ww-result-location"><strong>Selected location:</strong> {escape(state["location"] or "Not selected")}</p>'
            + '<p class="ww-result-bins"><strong>Available bins at this location:</strong> '
              + escape(', '.join(PILOT_BINS.get(state["location"], ())) or 'Select a location') + '</p>'
            + '<div class="ww-result-grid"><section class="ww-actions-copy">' + actions + '</section>' + review + '</div>'
            + education + '<footer class="ww-sources"><h3>Official UF guidance</h3><ul>' + "".join(links)
            + '</ul><details><summary>Sources and review details</summary>' + details + '</details></footer></article>')


def new_session():
    return {"step": 1, "location": None, "photo": None, "prediction": None, "item_id": None,
            "answers": {}, "question": None, "result": None, "result_back": None,
            "material_uncertain": False, "photo_message": "", "photo_feedback": None,
            "rejected_review": None,  # Recovery is only used by an explicit Back from material rejection.
            "decision_recorded": False}


def record_decision(state, withheld):
    # Keep this flag through navigation/rejection recovery; no identifiers are exported.
    if not state.get("decision_recorded", False):
        DECISION_COUNT.labels(outcome="withheld" if withheld else "guidance").inc()
        state["decision_recorded"] = True


def prediction_reviewable(state):
    prediction = state.get("prediction") or {}
    return (prediction.get("material") in classes
            and MIN_CONFIDENCE <= prediction.get("confidence", -1) <= 1
            and not state.get("material_uncertain"))


def prediction_confirmed(state):
    """Selecting a displayed item implicitly accepts a non-rejected material prediction."""
    return (prediction_reviewable(state)
            and state.get("item_id") in {key for _, key in item_choices(state["prediction"]["material"])})


def clear_photo(state, message="", photo_feedback=None):
    # Keep the location; invalidate the photo, recovery snapshot and all dependent information.
    state.update(step=2 if state["location"] in PILOT_LOCATIONS else 1,
                 photo=None, prediction=None, item_id=None, answers={}, question=None,
                 result=None, result_back=None, rejected_review=None,
                 material_uncertain=False, photo_message=message, photo_feedback=photo_feedback)
    return state


def show_decision(state, decision):
    if (decision["outcome"] in INFORMATION_OUTCOMES | {"ASK_CONDITION"}
            and not prediction_confirmed(state)):
        decision = {"outcome": "WITHHOLD", "message": "Select an item for the current prediction before continuing.",
                    "rule_id": None, "question": None}
    if decision["outcome"] == "ASK_CONDITION":
        state.update(step=4, question=decision["question"], result=None, result_back=None)
    else:
        record_decision(state, withheld=decision["outcome"] not in INFORMATION_OUTCOMES)
        # Count the decision event, never the repeated rendering of its result.
        if decision["outcome"] not in INFORMATION_OUTCOMES:
            WITHHOLDING_COUNT.inc()
        # Record the actual originating screen/question, including early condition outcomes.
        state["result_back"] = {"step": state["step"], "question": deepcopy(state["question"])}
        state.update(result=decision, question=None,
                     step=5 if decision["outcome"] in INFORMATION_OUTCOMES else 6)
    return state


def no_recommendation(state, message):
    return show_decision(state, {"outcome": "WITHHOLD", "message": message,
                                 "rule_id": None, "question": None})


def save_answer(state, answer):
    question = state["question"]
    if not question:
        return
    key = question["key"]
    selected = answer if answer in question["choices"] else None
    # Merely revisiting an unchanged answer must not discard still-applicable later answers.
    if selected == state["answers"].get(key):
        return
    keys = [q["key"] for q in questions_for(state["item_id"])]
    prior_keys = keys[:keys.index(key)]
    state["answers"] = {k: v for k, v in state["answers"].items() if k in prior_keys}
    if selected is not None:
        state["answers"][key] = selected
    state.update(result=None, result_back=None)


def advance(action, state, location=None, image_input=None, option_id=None, answer=None):
    """Context-aware navigation; only changed inputs invalidate their dependent state."""
    state = deepcopy(state)
    photo = image_input.get("path") if isinstance(image_input, dict) else image_input
    if action == "reset":
        return new_session()
    if action in ("photo_changed", "photo_cleared", "retry"):
        clear_photo(state)
        if action == "photo_changed":
            state["photo"] = photo
        return state
    if action == "location_changed" or (action == "next" and state["step"] == 1):
        selected = location if location in PILOT_LOCATIONS else None
        if selected != state["location"]:
            # Photo, item and condition facts are location-independent; rule results are not.
            state.update(location=selected, result=None, result_back=None, rejected_review=None)
        if action == "location_changed":
            state["step"] = 1
            return state
    if action == "uncertain":
        recovery = ({key: deepcopy(state[key]) for key in ("photo", "prediction", "item_id", "answers")}
                    if prediction_reviewable(state) else None)
        if recovery is not None:
            WITHHOLDING_COUNT.inc()
            record_decision(state, withheld=True)
        clear_photo(state, "The predicted material may be incorrect. Please upload or take another clear photo of a single item.")
        state.update(material_uncertain=True, rejected_review=recovery)
        return state
    if action == "back" and state["step"] == 2 and state["photo_message"]:
        recovery = state["rejected_review"] if state["material_uncertain"] else None
        clear_photo(state)
        if recovery and prediction_reviewable(recovery):
            # Explicitly cancel rejection; never restore a result or a below-threshold prediction.
            state.update(recovery)
            state["step"] = 3 if state["location"] in PILOT_LOCATIONS else 1
        return state
    if state["step"] in (3, 4, 5, 6) and not prediction_reviewable(state):
        return clear_photo(state, "The prediction cannot be used. Please submit another photo.")
    if state["step"] in (4, 5, 6) and not prediction_confirmed(state):
        return clear_photo(state, "The item selection is missing. Please submit another photo.")
    if state["step"] == 3 and action in ("review", "next", "back"):
        valid_options = {key for _, key in item_choices(state["prediction"]["material"])}
        selected = option_id if option_id in valid_options else None
        # Back preserves the stored choice when no control value was supplied.
        if action != "back" or option_id is not None:
            if selected != state["item_id"]:
                state.update(item_id=selected, answers={}, question=None, result=None, result_back=None)
        if action == "review" or (action == "next" and not prediction_confirmed(state)):
            return state
    if state["step"] == 4 and action in ("answer_changed", "back", "next"):
        if action == "back" and answer is None:
            answer = state["answers"].get(state["question"]["key"])
        save_answer(state, answer)
        if action == "answer_changed":
            return state
    if action == "back":
        if state["step"] == 4:
            questions = questions_for(state["item_id"])
            index = next(i for i, q in enumerate(questions) if q["key"] == state["question"]["key"])
            state.update(step=4 if index else 3,
                         question=questions[index - 1] if index else None,
                         result=None, result_back=None)
        elif state["step"] in (5, 6):
            origin = state["result_back"] or {"step": 3, "question": None}
            question = origin["question"]
            questions = questions_for(state["item_id"])
            if origin["step"] == 4 and question in questions:
                state.update(step=4, question=question)
            else:
                state.update(step=3, question=None)
            state.update(result=None, result_back=None)
        elif state["step"] == 3:
            state.update(step=2, question=None, result=None, result_back=None)
        elif state["step"] == 2:
            state.update(step=1, result=None, result_back=None)
        return state
    if action != "next":
        return state
    if state["step"] == 1:
        if state["location"] in PILOT_LOCATIONS:
            state["step"] = 2
    elif state["step"] == 2:
        # Warning screens require Back or Retry. A newly supplied photo invalidates the old one.
        if state["photo_message"] and photo is None:
            return state
        if photo is not None and photo != state["photo"]:
            clear_photo(state)
            state["photo"] = photo
        if state["photo"] is None:
            return state
        if prediction_reviewable(state):
            if not isinstance(state["photo"], (bytes, bytearray)) and not Path(state["photo"]).is_file():
                return clear_photo(state, "The uploaded image is unavailable. Please upload it again.")
            state.update(step=3, photo_message="", photo_feedback=None)
            return state  # Reuse eligible inference when the photo has not changed.
        photo = state["photo"]
        clear_photo(state)
        # Only a new inference starts a new attempt; Retry/Back/rendering do not.
        state["decision_recorded"] = False
        try:
            predicted = predict_material(photo)
            if not MIN_CONFIDENCE <= predicted["confidence"] <= 1:
                WITHHOLDING_COUNT.inc()
                record_decision(state, withheld=True)
                return clear_photo(state,
                    f"Softmax {predicted['confidence']:.2%} is below the provisional "
                    f"{MIN_CONFIDENCE:.0%} minimum or invalid. Try another clear photo of one item.",
                    photo_feedback={"material": predicted["material"], "confidence": predicted["confidence"]})
            state.update(photo=photo, prediction=predicted, step=3)
        except (ValueError, TypeError, OSError, RuntimeError) as error:
            return clear_photo(state, str(error) if isinstance(error, ValueError)
                               else "The photo could not be processed. Please try another image.")
    elif state["step"] == 3:
        # Revisit required questions in order, displaying any still-applicable saved answers.
        decision = rule_decision(state["prediction"]["material"], state["item_id"], {},
                                 state["location"], state["material_uncertain"])
        return show_decision(state, decision)
    elif state["step"] == 4:
        if answer not in state["question"]["choices"]:
            return state
        keys = [q["key"] for q in questions_for(state["item_id"])]
        through_current = keys[:keys.index(state["question"]["key"]) + 1]
        answers = {k: v for k, v in state["answers"].items() if k in through_current}
        decision = rule_decision(state["prediction"]["material"], state["item_id"], answers,
                                 state["location"], state["material_uncertain"])
        return show_decision(state, decision)
    return state


CSS = """
.gradio-container {
    width: calc(100% - 32px) !important; max-width: 820px !important; margin: 28px auto !important;
    flex: none !important; height: fit-content !important;
    padding: 24px !important; background: white; border: 1px solid #dce3e8;
    border-radius: 14px; box-shadow: 0 4px 24px #123b6008;
}
.gradio-container > .main {padding: 0 !important; flex: none !important;}
.ww-panel > button {align-self: flex-start; flex: none; width: auto;}
#brand {border-bottom: 1px solid #dce3e8; padding: 2px 0 16px; margin-bottom: 6px;}
.ww-brand {display: flex; align-items: center; gap: 12px;}
.ww-brand svg {width: 40px; height: 48px; flex-shrink: 0;}
.ww-brand > div {min-width: 0;}
.ww-brand h1 {display: flex; align-items: center; gap: 9px; margin: 0; color: #123b60;
    font: 750 34px/1.15 system-ui, sans-serif; letter-spacing: -1px;}
.ww-brand .ww-wordmark {color: #123b60;}
.ww-brand .ww-wise {color: #17613c;}
.ww-brand .ww-ai {padding: 4px 7px; border: 1px solid #c6d8ce; border-radius: 6px;
    background: #edf5f0; color: #17613c; font: 750 14px/1 system-ui, sans-serif; letter-spacing: .4px;}
.ww-brand p {margin: 5px 0 0; color: #435364; font: 13px/1.45 system-ui, sans-serif;}
.ww-panel {gap: 12px !important;}
.ww-panel h2 {font-size: 21px !important; line-height: 1.3 !important; margin: 0 0 6px !important; color: #123b60 !important;}
.ww-panel h3 {font-size: 19px !important; color: #123b60 !important;}
.ww-panel p {margin: 0 0 8px !important;}
#location-note p {font-size: 12px; color: #596775; margin: 0 !important;}
#location-picker {margin-top: 2px;}
.ww-actions {justify-content: space-between; gap: 10px !important; margin-top: 6px;}
.ww-actions button {flex: 0 1 auto !important; width: auto; min-width: 108px !important;}
.ww-actions button.primary {margin-left: auto;}
#start-over {align-self: flex-start; flex: none; width: auto; min-width: 0;
    padding: 4px 0; min-height: 30px; background: transparent; border: 0; box-shadow: none;
    color: #526374; font-size: 12px;}
#uncertain {align-self: stretch; width: 100%; flex: none; min-width: 0;
    padding: 12px 14px; border: 1px solid #c77b7b; border-radius: 10px;
    background: #fff0f0; color: #842d35; font-size: 14px; font-weight: 750;}
#uncertain:hover {background: #fbe2e4; border-color: #a64850;}
#uncertain:focus-visible {outline: 2px solid #842d35; outline-offset: 2px;}
.ww-warning.ww-material-rejected {background: #fff0f0; border: 2px solid #c77b7b; color: #652730;}
.ww-material-rejected .ww-warning-title h2 {color: #842d35 !important;}
.ww-material-rejected .ww-warning-title span {color: #842d35; font-size: 28px; line-height: 1;}
#progress {border-top: 1px solid #e6ebef; padding-top: 18px; margin-top: 4px;}
.ww-steps {display: flex; list-style: none; margin: 0; padding: 0; width: 100%;}
.ww-step {position: relative; flex: 1; min-width: 0; text-align: center; color: #687583; font: 12px/1.4 system-ui, sans-serif;}
.ww-step:not(:first-child)::before {content: ''; position: absolute; z-index: 0; top: 10px;
    left: calc(-50% + 11px); width: calc(100% - 22px); height: 2px; background: #d7dee4;}
.ww-circle {position: relative; z-index: 1; display: flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; box-sizing: border-box; margin: 0 auto 6px;
    border: 2px solid #c4cdd5; border-radius: 50%; background: white; font-size: 13px; font-weight: 700;}
.ww-step.complete, .ww-step.current {color: #17613c; font-weight: 650;}
.ww-step.complete .ww-circle, .ww-step.current .ww-circle {border-color: #17613c; background: #17613c; color: white;}
.ww-step.current .ww-circle {box-shadow: 0 0 0 3px #e1eee6;}
.ww-step.complete + .ww-step::before {background: #17613c;}

.ww-prediction {display: flex; align-items: center; justify-content: space-between; gap: 16px;
    padding: 16px; border: 1px solid #dce3e8; border-radius: 12px; background: #f8fafb;}
.ww-prediction .ww-material {font: 750 30px/1.15 system-ui, sans-serif !important;
    color: #123b60 !important; margin: 3px 0 0 !important; overflow-wrap: anywhere;}
.ww-eyebrow {display: block; font: 12px/1.4 system-ui, sans-serif; color: #596775;}
.ww-score {text-align: right; flex-shrink: 0;}
.ww-confidence {display: block; font: 800 34px/1.15 system-ui, sans-serif;}
.ww-confidence-pass {color: #17613c;}
.ww-confidence-low {color: #925b0b;}
.ww-score-note {font-size: 12px; color: #596775; margin-top: 7px !important;}
.ww-warning {padding: 18px; background: #fff6df; border: 1px solid #e7c16a;
    border-radius: 12px; color: #67440c;}
.ww-warning-title {display: flex; align-items: center; gap: 10px; margin-bottom: 8px;}
.ww-warning-title svg {width: 26px; height: 26px; flex-shrink: 0;}
.ww-warning-title h2 {color: #67440c !important; margin: 0 !important;}
.ww-warning .ww-prediction {background: #ffffffb3; border-color: #ead8ad; margin-top: 12px;}
.ww-warning ul {margin: 12px 0 0; padding-left: 20px; font-size: 14px; line-height: 1.7;}
#item-options .wrap {display: flex !important; flex-direction: column !important; gap: 8px !important;}
#item-options .wrap > label {display: flex !important; align-items: flex-start !important;
    justify-content: flex-start !important; gap: 10px; width: 100%; min-width: 0;
    box-sizing: border-box; margin: 0 !important; padding: 11px 13px !important;
    border: 1px solid #d7dee4; border-radius: 10px !important; text-align: left;
    white-space: normal; overflow-wrap: anywhere; background: white;}
#item-options .wrap > label:has(input:checked) {background: #e7f3eb; border-color: #8fbea0;}
#item-options .wrap > label:focus-within {outline: 2px solid #17613c; outline-offset: 2px;}
#item-options input[type="radio"] {flex: 0 0 17px; width: 17px; height: 17px;
    margin: 2px 0 0 !important; accent-color: #17613c;}

/* Only the below-threshold photo state uses these styles. */
.ww-low-confidence {display: grid; gap: 16px;}
.ww-low-confidence .ww-low-score {text-align: center; padding: 18px 16px;
    background: #fffcf5; border: 1px solid #ead8ad; border-radius: 12px;}
.ww-low-confidence .ww-low-score-label {display: block; font-size: 13px; color: #596775;}
.ww-low-confidence .ww-low-score-value {display: block; margin: 4px 0;
    font: 800 clamp(48px, 8vw, 60px)/1.1 system-ui, sans-serif; color: #925b0b;}
.ww-low-confidence .ww-low-material {font-size: 13px; color: #596775;}
.ww-low-confidence .ww-warning {border: 2px solid #d9a441; color: #46320e;}
.ww-low-confidence .ww-warning p {margin: 0 !important; line-height: 1.5;}
.ww-low-confidence .ww-low-tips h3 {font-size: 14px !important; margin: 0 0 6px !important;}
.ww-low-confidence .ww-low-tips ul {margin: 0; padding-left: 20px; line-height: 1.7; font-size: 14px;}
.ww-panel:has(.ww-low-confidence) .ww-actions button.primary {flex: 1 1 auto !important;
    min-height: 48px; padding: 12px 20px; font-size: 16px; font-weight: 750;}

@media (max-width: 600px) {
    .gradio-container {margin: 12px auto !important; padding: 16px !important; border-radius: 10px;}
    .ww-brand {gap: 10px;}
    .ww-brand svg {width: 32px; height: 40px;}
    .ww-brand h1 {font-size: 27px; gap: 7px; letter-spacing: -.8px;}
    .ww-brand .ww-ai {font-size: 12px; padding: 4px 6px;}
    .ww-prediction {padding: 12px; gap: 10px;}
    .ww-prediction .ww-material {font-size: 26px !important;}
    .ww-confidence {font-size: 30px;}
    .ww-warning {padding: 14px;}
    .ww-brand p {font-size: 12px;}
    .ww-panel h2 {font-size: 19px !important;}
    .ww-step {font-size: 10px;}
    .ww-actions button {min-width: 90px !important; font-size: 13px;}
}

/* Final outcomes share one layout; these styles do not affect earlier steps. */
#rule-info, #no-rule {border: 0; padding: 0;}
.ww-result {font: 16px/1.6 system-ui, sans-serif; color: #243746; overflow-wrap: anywhere;}
.ww-result h2 {margin-bottom: 12px !important;}
.ww-result h3 {font-size: 18px !important; margin: 0 0 12px !important;}
.ww-outcome {padding: 18px 20px; border: 1px solid; border-radius: 12px; margin-bottom: 12px;}
.ww-outcome-title {display: flex; align-items: center; gap: 12px;}
.ww-outcome-title > span {display: inline-flex; align-items: center; justify-content: center;
    flex-shrink: 0; width: 42px; height: 42px; border-radius: 50%; background: #ffffffb3;
    font-size: 30px; font-weight: 800; line-height: 1;}
.ww-outcome-title h3 {font-size: clamp(24px, 4vw, 30px) !important; line-height: 1.2; margin: 0 0 8px !important;}
.ww-outcome-recycle {background: #eaf5ee; border-color: #a3c8af; color: #17613c;}
.ww-outcome-recycle h3 {color: #17613c !important;}
.ww-outcome-caution {background: #fff6df; border-color: #dfbf75; color: #63460e;}
.ww-outcome-caution h3 {color: #63460e !important;}
.ww-outcome-exclude {background: #fff0e9; border-color: #dba58b; border-left: 5px solid #b14c30; color: #813a25;}
.ww-outcome-exclude h3 {color: #813a25 !important;}
.ww-destination {margin-top: 16px; padding: 14px 16px; background: #ffffffdb;
    border: 1px solid #00000025; border-radius: 9px;}
.ww-destination span {display: block; font-size: 13px; font-weight: 650; color: #435364;}
.ww-destination strong {display: block; font-size: clamp(24px, 4vw, 28px); line-height: 1.3; margin-top: 5px;}
.ww-outcome .ww-destination-note {font-size: 14px; line-height: 1.6; margin: 12px 0 0 !important;}
.ww-result-location {margin-bottom: 4px !important; font-size: 14px;}
.ww-result-bins {margin: 0 0 16px !important; font-size: 14px; line-height: 1.6; color: #435364;}
.ww-result .ww-muted {color: #596775; font-size: 12px;}
.ww-result-grid {display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(0, 1fr); gap: 22px; margin: 20px 0;}
.ww-actions-copy > ol {padding-left: 32px; margin: 0; list-style: decimal;}
.ww-actions-copy > ol > li {padding: 12px 14px; margin-bottom: 12px; border: 1px solid #dce3e8;
    border-radius: 8px; background: #f6f8fa; color: #243746; font-size: 17px; line-height: 1.65;}
.ww-actions-copy > ol > li::marker {font-size: 20px; font-weight: 800; color: #123b60;}
.ww-item-review {padding: 16px; border-radius: 10px; background: #f6f8fa; align-self: start;}
.ww-item-review h3 {font-size: 14px !important;}
.ww-item-review dl {margin: 0;}
.ww-item-review dt {color: #596775; font-size: 12px; margin-top: 10px;}
.ww-item-review dd {font-size: 13px; font-weight: 650; margin: 2px 0 0;}
.ww-components {margin-top: 20px;}
.ww-component {padding: 10px 0; border-top: 1px solid #e1e7eb;}
.ww-component h4 {font-size: 16px; margin: 0 0 6px; color: #123b60;}
.ww-evidence {display: inline-block; margin-left: 4px; padding: 1px 6px; border-radius: 4px;
    background: #fff2cf; color: #67440c; font-size: 10px; font-weight: 500;}
.ww-component-hold {font-size: 12px; color: #596775; margin-top: 14px !important;}
.ww-fallback {padding: 14px; margin-top: 16px; background: #fff9ea; border: 1px solid #ead8ad; border-radius: 8px;}
.ww-fallback h3 {font-size: 14px !important;}
.ww-fallback p {font-size: 15px; line-height: 1.65;}
.ww-education {border-top: 1px solid #e1e7eb; padding-top: 14px; margin-top: 8px;}
.ww-education h3 {font-size: 14px !important;}
.ww-education p {font-size: 13px;}
.ww-sources {padding-top: 10px; font-size: 12px;}
.ww-sources h3 {font-size: 13px !important; margin-bottom: 5px !important;}
.ww-sources ul {display: flex; flex-wrap: wrap; gap: 4px 16px; list-style: none; margin: 0; padding: 0;}
.ww-sources li > span {display: none;}
.ww-sources a {color: #123b60; text-decoration: underline; text-underline-offset: 2px;}
.ww-sources details {margin-top: 10px; color: #596775;}
.ww-sources summary {cursor: pointer; color: #596775;}
.ww-sources details p {margin-top: 8px !important; white-space: pre-line;}
@media (max-width: 600px) {
    .ww-result-grid {grid-template-columns: minmax(0, 1fr); gap: 16px; margin: 16px 0;}
    .ww-outcome {padding: 14px;}
    .ww-destination strong {font-size: 24px;}
    .ww-outcome-title > span {font-size: 26px;}
    .ww-result .ww-item-review {padding: 13px;}
}

"""

# Keep the reference's white, blue, and green palette even in a dark-mode browser.
THEME = gr.themes.Soft(
    primary_hue="green", secondary_hue="blue", neutral_hue="slate",
    spacing_size="sm", text_size="sm", font=["system-ui", "sans-serif"],
).set(**{
    key + suffix: value
    for key, value in {
        "body_background_fill": "#f5f7f9", "body_text_color": "#243746",
        "body_text_color_subdued": "#596775", "background_fill_primary": "white",
        "background_fill_secondary": "#f5f7f9", "block_background_fill": "white",
        "block_border_color": "#d7dee4", "block_label_background_fill": "white",
        "block_label_text_color": "#243746", "block_title_text_color": "#123b60",
        "input_background_fill": "white", "input_background_fill_focus": "white",
        "input_background_fill_hover": "#f5f7f9", "input_border_color": "#c4cdd5",
        "input_placeholder_color": "#687583", "border_color_primary": "#d7dee4",
        "button_primary_background_fill": "#17613c", "button_primary_background_fill_hover": "#104c2e",
        "button_primary_border_color": "#17613c", "button_primary_text_color": "white",
        "button_secondary_background_fill": "white", "button_secondary_background_fill_hover": "#f0f5f8",
        "button_secondary_border_color": "#c4cdd5", "button_secondary_text_color": "#123b60",
        "checkbox_label_background_fill": "white", "checkbox_label_background_fill_hover": "#f5f7f9",
        "checkbox_label_background_fill_selected": "#e7f3eb", "checkbox_label_text_color": "#243746",
        "checkbox_label_text_color_selected": "#17613c", "checkbox_label_border_color": "#d7dee4",
        "checkbox_label_border_color_selected": "#8fbea0", "link_text_color": "#123b60",
    }.items() for suffix in ("", "_dark")
})


def progress_html(state):
    """Render navigation status without marking skipped/withheld stages complete."""
    current = min(state["step"], 5)
    completed = {
        1: state["location"] in PILOT_LOCATIONS,
        2: state["prediction"] is not None,
        3: state["item_id"] not in (None, OTHER) and prediction_confirmed(state),
        4: prediction_confirmed(state) and bool(state["answers"]) and bool(state["result"])
           and state["result"]["outcome"] in INFORMATION_OUTCOMES,
    }
    steps = []
    for i, name in enumerate(["Location", "Photo", "Item", "Conditions", "Result"], 1):
        status = "current" if i == current else ("complete" if i < current and completed.get(i)
                  else "incomplete" if i < current else "upcoming")
        marker = "✓" if status == "complete" else ""
        current_attr = ' aria-current="step"' if status == "current" else ""
        steps.append(f'<li class="ww-step {status}"{current_attr} aria-label="{name}: {status}">'
                     f'<span class="ww-circle" aria-hidden="true">{marker}</span><span>{name}</span></li>')
    return '<nav aria-label="Disposal progress"><ol class="ww-steps">' + "".join(steps) + '</ol></nav>'


# Gradio 6.31 Dropdown has no placeholder parameter. This changes only its
# displayed prompt; None stays the actual value and no extra choice is added.
LOCATION_PROMPT_JS = """() => {
    const picker = document.getElementById('location-picker');
    if (!picker) return;
    const setPrompt = () => {
        const input = picker.querySelector('input');
        if (input && input.placeholder !== 'Select a location') {
            input.placeholder = 'Select a location';
        }
    };
    setPrompt();
    new MutationObserver(setPrompt).observe(picker, {
        childList: true, subtree: true, attributes: true, attributeFilter: ['placeholder']
    });
}"""


def prediction_html(predicted):
    """Present the actual score; coloring never changes acceptance or rule decisions."""
    if not predicted:
        return ""
    score = predicted["confidence"]
    valid = 0 <= score <= 1
    score_class = "ww-confidence-pass" if valid and score >= MIN_CONFIDENCE else "ww-confidence-low"
    score_text = f"{score:.2%}" if valid else "Unavailable"
    return (
        '<div class="ww-prediction"><div><span class="ww-eyebrow">Predicted material</span>'
        f'<h3 class="ww-material">{escape(predicted["material"].title())}</h3></div>'
        '<div class="ww-score"><span class="ww-eyebrow">Confidence</span>'
        f'<strong class="ww-confidence {score_class}">{score_text}</strong></div></div>'
        '<p class="ww-score-note">A high score can still be wrong.</p>'
    )


def photo_warning_html(state):
    if not state["photo_message"]:
        return ""
    feedback = state.get("photo_feedback")
    title = "We need a clearer photo." if feedback else "Please try another photo."
    icon = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2 23 21H1Z" '
            'fill="#f2c14e" stroke="#925b0b" stroke-width="1.5"/>'
            '<path d="M12 8v6" stroke="#67440c" stroke-width="2" stroke-linecap="round"/>'
            '<circle cx="12" cy="17.5" r="1.2" fill="#67440c"/></svg>')
    if feedback and 0 <= feedback["confidence"] < MIN_CONFIDENCE:
        # Presentation only: the existing gate has already cleared the usable prediction.
        return (
            '<section class="ww-low-confidence"><h2>We need a clearer photo.</h2>'
            '<div class="ww-low-score"><span class="ww-low-score-label">Model confidence</span>'
            f'<strong class="ww-low-score-value">{feedback["confidence"]:.2%}</strong>'
            f'<span class="ww-low-material">Predicted material: {escape(feedback["material"].title())}</span></div>'
            f'<div class="ww-warning" role="alert"><div class="ww-warning-title">{icon}'
            '<h2>Low confidence prediction</h2></div>'
            f'<p>This photo is below the provisional {MIN_CONFIDENCE:.0%} minimum confidence threshold. '
            'Please try another clear photo of one item.</p></div>'
            '<div class="ww-low-tips"><h3>Tips for your next photo</h3>'
            '<ul><li>Use good lighting.</li><li>Make sure the item is in focus.</li>'
            '<li>Include only one item.</li></ul></div></section>'
        )
    if state["material_uncertain"]:
        return ('<div class="ww-warning ww-material-rejected" role="alert">'
                '<div class="ww-warning-title"><span aria-hidden="true">⚠</span>'
                '<h2>Material not confirmed</h2></div>'
                f'<p>{escape(state["photo_message"])}</p></div>')
    if feedback:
        message = (f'The prediction is below the {MIN_CONFIDENCE:.0%} review threshold.'
                   if 0 <= feedback["confidence"] < MIN_CONFIDENCE
                   else "The prediction could not be used. Please take another photo.")
        details = prediction_html(feedback) + (
            '<ul><li>Use good lighting.</li><li>Make sure the item is in focus.</li>'
            '<li>Include only one item.</li></ul>')
    else:
        message, details = state["photo_message"], ""
    return (f'<div class="ww-warning" role="alert"><div class="ww-warning-title">{icon}'
            f'<h2>{title}</h2></div><p>{escape(message)}</p>{details}</div>')


def build_app():
    # Unclaimed uploads expire within two minutes; live-session files are protected.
    with gr.Blocks(title="WasteWise AI", analytics_enabled=False, delete_cache=(60, 60)) as app:
        session = gr.State(new_session())
        gr.HTML(
            '<header class="ww-brand">'
            '<svg class="ww-leaf" viewBox="0 0 40 48" aria-hidden="true">'
            '<defs><linearGradient id="ww-leaf-blue" x2="0" y2="1">'
            '<stop stop-color="#154e74"/><stop offset="1" stop-color="#062d49"/>'
            '</linearGradient><linearGradient id="ww-leaf-green" x1="0" y1="1" x2="1" y2="0">'
            '<stop stop-color="#087535"/><stop offset="1" stop-color="#58a433"/>'
            '</linearGradient></defs><g transform="translate(2 0) scale(.2)">'
            '<path fill="url(#ww-leaf-blue)" d="M176 1C153 21 124 24 89 37C32 58-8 111 18 183'
            'C36 137 63 117 108 94C154 72 173 43 176 1Z"/>'
            '<path fill="#062d49" opacity=".45" d="M176 1C161 37 137 53 92 70'
            'C38 91 9 127 18 183C36 137 63 117 108 94C154 72 173 43 176 1Z"/>'
            '<path fill="url(#ww-leaf-green)" d="M176 1C185 71 174 127 140 155'
            'C117 175 83 178 57 183C35 187 22 205 14 232C12 241-2 237 1 229'
            'C12 187 33 155 70 127C94 109 126 96 145 77C167 55 176 28 176 1Z"/>'
            '<path fill="#086532" opacity=".65" d="M176 1C176 28 167 55 145 77'
            'C126 96 94 109 70 127C33 155 12 187 1 229C16 190 41 168 80 156'
            'C143 136 174 87 176 1Z"/></g></svg>'
            '<div><h1><span class="ww-wordmark">Waste<span class="ww-wise">Wise</span></span> '
            '<span class="ww-ai">AI</span></h1>'
            '<p>Take a photo. Get guidance. Dispose correctly.</p></div></header>',
            elem_id="brand",
        )
        panels = []
        next_buttons = []
        back_buttons = []
        scan_buttons = []
        with gr.Column(visible=True, elem_classes="ww-panel") as panel:
            gr.Markdown("## Where are you disposing of your item?\nSelect your UF campus location.")
            location = gr.Dropdown(PILOT_LOCATIONS, label="UF campus location", value=None,
                                   show_label=False, filterable=False, interactive=True,
                                   elem_id="location-picker")
            available_bins = gr.Markdown(location_note(None), elem_id="location-note")
            with gr.Row(elem_classes="ww-actions"):
                location_next = gr.Button("Next →", variant="primary", interactive=False, elem_id="location-next")
                next_buttons.append(location_next)
        panels.append(panel)
        with gr.Column(visible=False, elem_classes="ww-panel") as panel:
            photo_heading = gr.Markdown("## Add a photo of your item\nShow one whole item in good lighting against a plain background.")
            photo_message = gr.HTML(visible=False)
            with gr.Column() as photo_entry:
                upload = gr.Image(type="filepath", image_mode=None, sources=["upload", "webcam"],
                                  label="Item photo — JPEG, PNG, or WebP; maximum 5 MB", height=240,
                                  buttons=[], format="png")
            with gr.Row(elem_classes="ww-actions"):
                back_buttons.append(gr.Button("← Back"))
                photo_next = gr.Button("Next →", variant="primary", interactive=False, elem_id="photo-next")
                retry = gr.Button("Try another photo", variant="primary", visible=False)
                next_buttons.append(photo_next)
        panels.append(panel)
        with gr.Column(visible=False, elem_classes="ww-panel") as panel:
            gr.Markdown("## Review material and select item")
            prediction = gr.HTML()
            gr.Markdown("Make sure the predicted material is correct and the photo shows only one item.")
            item = gr.Radio(choices=[], label="Which item is it?", value=None,
                            interactive=False, visible=False, elem_id="item-options")
            uncertain = gr.Button("⚠ Material is uncertain / looks wrong", elem_id="uncertain")
            with gr.Row(elem_classes="ww-actions"):
                back_buttons.append(gr.Button("← Back — change photo"))
                item_next = gr.Button("Next →", variant="primary", interactive=False, elem_id="item-next")
                next_buttons.append(item_next)
        panels.append(panel)
        with gr.Column(visible=False, elem_classes="ww-panel") as panel:
            gr.Markdown("## We need a little more information")
            condition_summary = gr.Markdown()
            answer = gr.Radio(choices=[], label="Condition question", value=None, interactive=True)
            gr.Markdown("Answer each required question. Not sure withholds guidance.")
            with gr.Row(elem_classes="ww-actions"):
                back_buttons.append(gr.Button("← Back"))
                condition_next = gr.Button("Next →", variant="primary", interactive=False, elem_id="condition-next")
                next_buttons.append(condition_next)
        panels.append(panel)
        with gr.Column(visible=False, elem_classes="ww-panel") as panel:
            guidance = gr.HTML(elem_id="rule-info")
            with gr.Row(elem_classes="ww-actions"):
                back_buttons.append(gr.Button("Back — review answers"))
                scan_buttons.append(gr.Button("📷 Scan another item", variant="primary", elem_id="scan-result"))
        panels.append(panel)
        with gr.Column(visible=False, elem_classes="ww-panel") as panel:
            withheld = gr.HTML(elem_id="no-rule")
            with gr.Row(elem_classes="ww-actions"):
                back_buttons.append(gr.Button("Back — review selections or answers"))
                scan_buttons.append(gr.Button("📷 Scan another item", variant="primary", elem_id="scan-withheld"))
        panels.append(panel)
        reset = gr.Button("Start Over", visible=False, elem_id="start-over")
        progress = gr.HTML(progress_html(new_session()), elem_id="progress")
        app.load(fn=None, js=LOCATION_PROMPT_JS)


        outputs = [session, progress, reset, *panels, prediction, item, condition_summary,
                   answer, guidance, withheld, location, upload, photo_message,
                   item_next, location_next, condition_next,
                   photo_heading, photo_entry, photo_next, retry, available_bins]

        # Gradio deduplicates identical uploads. Track ownership so closing one
        # tab cannot delete an image still used by another live tab.
        session_uploads = {}
        upload_lock = Lock()

        def track_upload(image_input, request):
            if request is None or not request.session_hash or image_input is None:
                return
            path = image_input.get("path") if isinstance(image_input, dict) else image_input
            if not isinstance(path, (str, Path)):
                raise gr.Error("Please upload the image again.", print_exception=False)
            path = Path(path).resolve()
            if not path.is_relative_to(UPLOAD_ROOT) or not path.is_file():
                raise gr.Error("Please upload the image again.", print_exception=False)
            with upload_lock:
                session_uploads.setdefault(request.session_hash, set()).add(str(path))
                upload.keep_in_cache.add(str(path))

        def cleanup_session(request: gr.Request):
            if request is None or not request.session_hash:
                return
            with upload_lock:
                paths = session_uploads.pop(request.session_hash, set())
                in_use = set().union(*session_uploads.values()) if session_uploads else set()
                for name in paths - in_use:
                    upload.keep_in_cache.discard(name)
                    path = Path(name)
                    # Delete only files in this app's private cache, never source files.
                    if not path.resolve().is_relative_to(UPLOAD_ROOT):
                        continue
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        continue  # Leave it tracked for scheduled cache cleanup.
                    for files in app.temp_file_sets:
                        files.discard(name)
                    try:
                        path.parent.rmdir()  # Only remove an empty hash directory.
                    except OSError:
                        pass

        app.unload(cleanup_session)

        def respond(action, state, selected_location, option_id, selected_answer, image_input=None,
                    request: gr.Request = None):
            track_upload(image_input, request)
            if action == "next" and state["step"] == 2 and image_input is None and not state["photo_message"]:
                state = advance("photo_cleared", state)  # A missing UI upload cannot reuse cached inference.
            state = advance(action,
                            state, selected_location, image_input, option_id, selected_answer)
            ready = prediction_reviewable(state)
            predicted = state["prediction"]
            material = predicted["material"] if predicted else ""
            question = state["question"]
            option = item_options.get(state["item_id"], {})
            warning = state["step"] == 2 and bool(state["photo_message"])
            update = {
                session: state, progress: progress_html(state),
                available_bins: location_note(state["location"]),
                reset: gr.Button(visible=state["step"] != 1),
                prediction: prediction_html(predicted),
                photo_message: gr.HTML(value=photo_warning_html(state), visible=warning),
                photo_heading: gr.Markdown(visible=not warning),
                photo_entry: gr.Column(visible=not warning),
                photo_next: gr.Button(visible=not warning, interactive=state["photo"] is not None),
                location_next: gr.Button(interactive=state["location"] in PILOT_LOCATIONS),
                condition_next: gr.Button(interactive=bool(question) and prediction_confirmed(state)
                                          and state["answers"].get(question["key"]) in question["choices"]),
                retry: gr.Button(visible=warning),
                item_next: gr.Button(interactive=prediction_confirmed(state)),
                item: gr.Radio(choices=item_choices(material) if ready else [],
                               value=state["item_id"] if ready else None, visible=ready, interactive=ready),
                condition_summary: "**Selected item:** " + option.get("user-facing option (English)", ""),
                answer: gr.Radio(choices=question["choices"] if question else [],
                                 label=question["text"] if question else "Condition question",
                                 value=state["answers"].get(question["key"]) if question else None),
                guidance: result_text(state) if state["step"] == 5 else "",
                withheld: result_text(state) if state["step"] == 6 else "",
            }
            update.update({panel: gr.Column(visible=state["step"] == i) for i, panel in enumerate(panels, 1)})
            update[location] = state["location"]
            # Retain the browser upload on upload events; restore it explicitly when navigating.
            if action != "photo_changed":
                update[upload] = state["photo"]
            return update

        inputs = [session, location, item, answer]
        for component, action in [(item, "review"), (location, "location_changed"),
                                  (answer, "answer_changed")]:
            component.input(partial(respond, action), inputs=inputs, outputs=outputs,
                            api_visibility="private", concurrency_limit=1,
                            concurrency_id="workflow", trigger_mode="always_last")
        # Upload and removal are separate events: a clear also emits .input with the old value.
        upload.upload(partial(respond, "photo_changed"), inputs=inputs + [upload], outputs=outputs,
                      preprocess=False, api_visibility="private", concurrency_limit=1,
                      concurrency_id="workflow", trigger_mode="always_last")
        upload.clear(partial(respond, "photo_cleared"), inputs=inputs, outputs=outputs,
                     api_visibility="private", concurrency_limit=1,
                     concurrency_id="workflow", trigger_mode="once")
        for action, buttons in [("next", next_buttons), ("back", back_buttons),
                                ("reset", [reset]), ("uncertain", [uncertain]), ("retry", [retry, *scan_buttons])]:
            for button in buttons:
                button.click(partial(respond, action),
                             inputs=inputs + [upload] if button is photo_next else inputs, outputs=outputs,
                             preprocess=button is not photo_next,
                             api_visibility="private", concurrency_limit=1,
                             concurrency_id="workflow", trigger_mode="once")
    return app




def launch_app(app):
    # Start only when launching the UI; importing the module opens no listener.
    metrics_server, metrics_thread = start_http_server(8000, addr=HOST, registry=METRICS_REGISTRY)
    try:
        return app.launch(
            server_name=HOST, server_port=PORT, share=False, inbrowser=False, inline=False,
            prevent_thread_lock=False, max_file_size=MAX_UPLOAD_BYTES, show_error=False,
            enable_monitoring=False, run_history=False, footer_links=[],
            theme=THEME,
            css=CSS,
        )
    finally:
        metrics_server.shutdown()
        metrics_server.server_close()
        metrics_thread.join()


if __name__ == "__main__":
    # A normal service stop must also run the upload-cache cleanup below.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    demo = build_app()
    try:
        launch_app(demo)
    finally:
        demo.close()
        UPLOAD_CACHE.cleanup()

