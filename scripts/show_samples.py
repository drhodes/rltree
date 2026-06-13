"""
Print a handful of pretty-printed synthetic dataset SVGs so you can see
what the generator is producing.
"""

import xml.dom.minidom

from rltree.dataset import generate_random_svg

N = 5
print(f"=== {N} Random Dataset SVGs ===")
for i in range(N):
    svg = generate_random_svg(seed=i)
    pretty = xml.dom.minidom.parseString(svg).toprettyxml(indent="  ")
    # Strip the XML declaration line
    body = "\n".join(ln for ln in pretty.splitlines() if not ln.startswith("<?xml"))
    print(f"\n--- Sample {i + 1} (seed={i}) ---")
    print(body)
