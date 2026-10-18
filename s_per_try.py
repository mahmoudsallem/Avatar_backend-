"""Server-side throughput from the server log (independent of upload speed).
Usage: python s_per_try.py server.log [--last N]   (N = number of TIMING lines = one benchmark batch, default all)
Prints: avatars, tries, server wall, avatars/min, seconds of server time per try."""
import re, sys
from datetime import datetime, timedelta
rx = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+).*TIMING .*?tries=(\d+) total_server=([\d.]+)s")
rows = []
for line in open(sys.argv[1], errors="ignore"):
    m = rx.search(line)
    if m:
        end = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") + timedelta(milliseconds=int(m.group(2)))
        rows.append((end - timedelta(seconds=float(m.group(4))), end, int(m.group(3))))
if "--last" in sys.argv:
    rows = rows[-int(sys.argv[sys.argv.index("--last") + 1]):]
if not rows:
    sys.exit("no TIMING lines found")
start, end = min(r[0] for r in rows), max(r[1] for r in rows)
wall = (end - start).total_seconds(); tries = sum(r[2] for r in rows)
print(f"avatars={len(rows)} tries={tries} ({tries/len(rows):.2f}/avatar) server_wall={wall:.1f}s "
      f"-> {len(rows)*60/wall:.1f} avatars/min, {wall/tries:.2f} s per try")
