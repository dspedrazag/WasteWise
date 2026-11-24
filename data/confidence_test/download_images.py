"""Download Commons renditions of the sources in labels.csv; preserve provenance."""
import csv
import hashlib
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from PIL import Image

ROOT = Path(__file__).resolve().parent
API = 'https://commons.wikimedia.org/w/api.php'
HEADERS = {'User-Agent': 'WasteWise-educational-confidence-test/1.0 (local academic project)'}


def fetch(url):
    for attempt in range(3):
        # Stay below the anonymous request rate and honor server back-off.
        time.sleep(7)
        try:
            with urlopen(Request(url, headers=HEADERS), timeout=90) as response:
                return response.read()
        except HTTPError as error:
            if error.code not in (429, 503) or attempt == 2:
                raise
            retry_after = error.headers.get('Retry-After', '')
            if retry_after.isdigit():
                delay = int(retry_after)
            elif retry_after:
                delay = max(0, (parsedate_to_datetime(retry_after)
                                - datetime.now(timezone.utc)).total_seconds())
            else:
                delay = 60 * (attempt + 1)
            print(f'HTTP {error.code}; waiting {delay:.0f}s before retrying {url}', flush=True)
            time.sleep(delay)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))


def clean(value):
    return html.unescape(re.sub(r'<[^>]*>', ' ', value)).strip()


with (ROOT / 'labels.csv').open(encoding='utf-8', newline='') as file:
    rows = list(csv.DictReader(file))

out = ROOT / 'images'
out.mkdir(exist_ok=True)
previous = {}
if (ROOT / 'sources.csv').exists():
    with (ROOT / 'sources.csv').open(encoding='utf-8', newline='') as file:
        previous = {row['filename']: row for row in csv.DictReader(file)}
records = []
for row in rows:
    record = dict(row)
    try:
        target = out / row['filename']
        saved = previous.get(row['filename'], {})
        if target.exists():
            data = target.read_bytes()
            if (saved.get('download_status') != 'ok'
                    or saved.get('sha256') != hashlib.sha256(data).hexdigest()
                    or any(saved.get(key) != value for key, value in row.items())):
                raise ValueError('Existing image/provenance mismatch; not overwriting')
            with Image.open(target) as image:
                image.verify()
            records.append(saved)
            print('VERIFIED', row['filename'], flush=True)
            continue
        requested_width = 1280
        params = {'action': 'query', 'titles': row['commons_title'],
                  'prop': 'imageinfo', 'iiprop': 'url|size|extmetadata|timestamp|sha1',
                  'iiurlwidth': requested_width, 'format': 'json', 'formatversion': '2'}
        page = json.loads(fetch(API + '?' + urlencode(params)))['query']['pages'][0]
        info = page['imageinfo'][0]
        # Small sources otherwise return the original URL, which returned HTTP 429.
        if info['width'] <= requested_width:
            requested_width = max(1, info['width'] // 2)
            params['iiurlwidth'] = requested_width
            page = json.loads(fetch(API + '?' + urlencode(params)))['query']['pages'][0]
            info = page['imageinfo'][0]
        metadata = info.get('extmetadata', {})
        for key, field in [('Artist', 'author'), ('LicenseShortName', 'license'),
                           ('LicenseUrl', 'license_url'), ('DateTimeOriginal', 'date_taken'),
                           ('Attribution', 'attribution'), ('Credit', 'credit')]:
            record[field] = clean(metadata.get(key, {}).get('value', ''))
        download_url = info.get('thumburl', info['url'])
        is_thumbnail = download_url != info['url']
        record.update(original_url=info['url'], download_url=download_url,
                      upload_timestamp=info.get('timestamp', ''),
                      transformation=(f'Wikimedia resized rendition, requested width {requested_width}px; no local edits'
                                      if is_thumbnail else 'none; original file bytes'), sha256='')
        data = fetch(download_url)
        import io
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        sha1 = info.get('sha1', '')
        if not is_thumbnail and re.fullmatch(r'[0-9a-f]{40}', sha1) and hashlib.sha1(data).hexdigest() != sha1:
            raise ValueError('File checksum differs from Commons; not overwriting')
        if not target.exists():
            target.write_bytes(data)
        record.update(sha256=hashlib.sha256(data).hexdigest(), download_status='ok', error='')
        print('OK', row['filename'], flush=True)
    except Exception as exc:
        record.update(download_status='failed', error=str(exc))
        print('FAILED', row['filename'], str(exc), flush=True)
    records.append(record)
    time.sleep(0.5)

fields = list(rows[0]) + ['author','license','license_url','date_taken','attribution','credit',
                         'original_url','download_url','upload_timestamp','transformation','sha256','download_status','error']
with (ROOT / 'sources.csv').open('w', encoding='utf-8', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
failed = sum(r['download_status'] != 'ok' for r in records)
print(f'{len(records)-failed}/{len(records)} images ready; source metadata in sources.csv.')
if failed:
    raise SystemExit('Some downloads failed. Do not silently evaluate a smaller sample.')
