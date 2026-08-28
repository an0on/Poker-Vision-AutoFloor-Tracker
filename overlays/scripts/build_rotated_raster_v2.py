from pathlib import Path
import base64, json, math

src = Path('/opt/data/cache/images/img_4f4b3352d112.jpg')
svg_path = Path('/opt/data/tmp/poker_table_raster_v3_rotated.svg')
json_path = Path('/opt/data/tmp/poker_table_raster_v3_rotated.json')
img_b64 = base64.b64encode(src.read_bytes()).decode('ascii')

# Original image size (landscape)
W, H = 1280, 719
# Rotated clockwise canvas (portrait)
RW, RH = H, W
transform = f'translate({H},0) rotate(90)'

# More table-design-aligned geometry in ORIGINAL image coordinates
# Tuned against the real printed lines rather than only the rough hand sketch.
outer = {'x': 26, 'y': 50, 'w': 1222, 'h': 620, 'rx': 175, 'ry': 175}
inner_rail = {'x': 66, 'y': 94, 'w': 1142, 'h': 545, 'rx': 155, 'ry': 155}
action = {'x': 215, 'y': 224, 'w': 850, 'h': 272, 'rx': 132, 'ry': 132}

# Divider lines tuned to the actual printed wedge boundaries.
lines = {
    'left_mid': ((215, 357), (81, 357)),
    'top_1': ((328, 244), (223, 112)),
    'top_2': ((474, 230), (398, 112)),
    'top_3': ((639, 225), (637, 104)),
    'top_4': ((806, 229), (851, 112)),
    'top_5': ((1000, 244), (1089, 112)),
    'right_mid': ((1064, 357), (1211, 357)),
    'bot_1': ((320, 472), (218, 610)),
    'bot_2': ((547, 489), (500, 613)),
    'bot_3': ((806, 489), (849, 614)),
    'bot_4': ((995, 472), (1073, 612)),
}

# Precise board grouping over the printed 5 card boxes.
board = {
    'flop_group': {'x': 520, 'y': 279, 'w': 138, 'h': 92},
    'turn': {'x': 664, 'y': 279, 'w': 46, 'h': 92},
    'river': {'x': 716, 'y': 279, 'w': 46, 'h': 92},
    'board_group_outline': {'x': 508, 'y': 268, 'w': 266, 'h': 110}
}

# Seat wedges fitted to the printed blue/black wedge pattern.
seat_polys = {
    '1':  [(639,104),(851,112),(806,229),(639,225)],
    '2':  [(851,112),(1089,112),(1000,244),(806,229)],
    '3':  [(1089,112),(1208,95),(1211,357),(1064,357),(1000,244)],
    '4':  [(1064,357),(1211,357),(1208,639),(1073,612),(995,472)],
    '5':  [(849,614),(1073,612),(995,472),(806,489)],
    '6':  [(637,615),(849,614),(806,489),(639,494)],
    '7':  [(500,613),(637,615),(639,494),(547,489)],
    '8':  [(218,610),(500,613),(547,489),(320,472)],
    '9':  [(66,639),(218,610),(320,472),(215,357),(66,357)],
    '10': [(66,94),(223,112),(328,244),(215,357),(66,357)]
}

def racetrack_path(x, y, w, h, rx=None, ry=None):
    r = min(h / 2.0, rx if rx is not None else h / 2.0)
    x0 = x
    x1 = x + w
    y0 = y
    y1 = y + h
    return (
        f'M {x0+r:.2f},{y0:.2f} '
        f'L {x1-r:.2f},{y0:.2f} '
        f'A {r:.2f},{r:.2f} 0 0 1 {x1-r:.2f},{y1:.2f} '
        f'L {x0+r:.2f},{y1:.2f} '
        f'A {r:.2f},{r:.2f} 0 0 1 {x0+r:.2f},{y0:.2f} Z'
    )

# (x, y) -> rotated clockwise coords
# in rotated canvas: x' = H - y, y' = x
def rot(pt):
    x, y = pt
    return [round(H - y, 2), round(x, 2)]

def rect_to_rot_norm(r):
    pts = [rot((r['x'], r['y'])), rot((r['x']+r['w'], r['y'])), rot((r['x']+r['w'], r['y']+r['h'])), rot((r['x'], r['y']+r['h']))]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return {'x': round(x1/RW,4), 'y': round(y1/RH,4), 'w': round((x2-x1)/RW,4), 'h': round((y2-y1)/RH,4)}

def poly_to_rot_norm(poly):
    pts = [rot(p) for p in poly]
    return [[round(x/RW,4), round(y/RH,4)] for x, y in pts]

seat_anchors = {}
for sid, poly in seat_polys.items():
    rpts = [rot(p) for p in poly]
    rx = sum(p[0] for p in rpts) / len(rpts)
    ry = sum(p[1] for p in rpts) / len(rpts)
    seat_anchors[sid] = {'x': round(rx/RW,4), 'y': round(ry/RH,4)}

with svg_path.open('w', encoding='utf-8') as f:
    f.write(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{RW}" height="{RH}" viewBox="0 0 {RW} {RH}">
  <defs>
    <style>
      .outer {{ fill: none; stroke: #ff2a2a; stroke-width: 7; }}
      .inner {{ fill: none; stroke: #ff2a2a; stroke-width: 5; opacity: 0.95; }}
      .seatfill {{ fill: rgba(255,230,0,0.11); stroke: none; }}
      .divider {{ stroke: #ffe600; stroke-width: 4.5; stroke-linecap: round; }}
      .action {{ fill: rgba(255,140,0,0.12); stroke: #ffe066; stroke-width: 4.5; }}
      .boardgroup {{ fill: none; stroke: #25ff52; stroke-width: 3.5; }}
      .board {{ fill: rgba(37,255,82,0.12); stroke: #25ff52; stroke-width: 3; }}
      .label {{ font: 22px Arial, sans-serif; fill: #ffffff; font-weight: bold; }}
      .slabel {{ font: 18px Arial, sans-serif; fill: #ffe600; font-weight: bold; }}
    </style>
  </defs>
  <image href="data:image/jpeg;base64,{img_b64}" x="0" y="0" width="{W}" height="{H}" transform="{transform}"/>
  <g transform="{transform}">
    <path class="outer" d="{racetrack_path(**outer)}"/>
    <path class="inner" d="{racetrack_path(**inner_rail)}"/>
''')
    for sid, poly in seat_polys.items():
        pts = ' '.join(f'{x},{y}' for x, y in poly)
        cx = sum(p[0] for p in poly)/len(poly)
        cy = sum(p[1] for p in poly)/len(poly)
        f.write(f'    <polygon class="seatfill" points="{pts}"/>\n')
        f.write(f'    <text class="slabel" x="{cx-6:.1f}" y="{cy:.1f}">{sid}</text>\n')
    for p1, p2 in lines.values():
        f.write(f'    <line class="divider" x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}"/>\n')
    f.write(f'    <path class="action" d="{racetrack_path(**action)}"/>\n')
    f.write(f'    <rect class="boardgroup" x="{board["board_group_outline"]["x"]}" y="{board["board_group_outline"]["y"]}" width="{board["board_group_outline"]["w"]}" height="{board["board_group_outline"]["h"]}" rx="8"/>\n')
    for key in ['flop_group', 'turn', 'river']:
        r = board[key]
        f.write(f'    <rect class="board" x="{r["x"]}" y="{r["y"]}" width="{r["w"]}" height="{r["h"]}" rx="6"/>\n')
    f.write('  </g>\n</svg>\n')

out = {
    'table_model_id': 'dopo_10max_v3_rotated_precise_from_user_annotations',
    'source_image': str(src),
    'orientation': 'rotated_90_clockwise',
    'rotated_canvas': {'width': RW, 'height': RH},
    'global_zones': {
        'outer_rail_outer': rect_to_rot_norm(outer),
        'outer_rail_inner': rect_to_rot_norm(inner_rail),
        'action_area': rect_to_rot_norm(action),
        'board_flop_group': rect_to_rot_norm(board['flop_group']),
        'board_turn': rect_to_rot_norm(board['turn']),
        'board_river': rect_to_rot_norm(board['river']),
        'board_group_outline': rect_to_rot_norm(board['board_group_outline']),
        'dealer_button_path_boundary_note': 'Dealer button path follows the boundary between seat wedges and the action area.'
    },
    'seat_divider_lines': {
        name: {
            'p1': [round(rot(p1)[0]/RW,4), round(rot(p1)[1]/RH,4)],
            'p2': [round(rot(p2)[0]/RW,4), round(rot(p2)[1]/RH,4)]
        }
        for name, (p1, p2) in lines.items()
    },
    'seats': {}
}
for sid, poly in seat_polys.items():
    out['seats'][sid] = {
        'anchor': seat_anchors[sid],
        'seat_wedge_polygon': poly_to_rot_norm(poly),
        'occupancy_priority': ['chips', 'all_in_button', 'face_down_cards_secondary'],
        'notes': 'Seat wedge tuned to the printed table pattern and the user-drawn segmentation.'
    }
json_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
print(svg_path)
print(json_path)
