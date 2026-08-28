from pathlib import Path
import base64, json

src = Path('/opt/data/cache/images/img_4f4b3352d112.jpg')
svg_path = Path('/opt/data/tmp/poker_table_calibration_instance_current_table_v1_landscape.svg')
png_path = Path('/opt/data/tmp/poker_table_calibration_instance_current_table_v1_landscape.png')
json_path = Path('/opt/data/tmp/poker_table_calibration_instance_current_table_v1_landscape.json')
img_b64 = base64.b64encode(src.read_bytes()).decode('ascii')

W, H = 1280, 719

# Final landscape-oriented calibration instance based on the user's annotated image
outer = {
    'top_left': [201, 50],
    'top_right': [1078, 50],
    'bottom_right': [1078, 670],
    'bottom_left': [201, 670],
}
inner = {
    'top_left': [221, 94],
    'top_right': [1060, 94],
    'bottom_right': [1060, 639],
    'bottom_left': [221, 639],
}
action = {
    'top_left': [351, 224],
    'top_right': [930, 224],
    'bottom_right': [930, 496],
    'bottom_left': [351, 496],
}

seat_dividers = {
    'd1':  {'outer_ring_point': [223, 112], 'inner_ring_point': [328, 244]},
    'd2':  {'outer_ring_point': [398, 112], 'inner_ring_point': [474, 230]},
    'd3':  {'outer_ring_point': [637, 104], 'inner_ring_point': [639, 225]},
    'd4':  {'outer_ring_point': [851, 112], 'inner_ring_point': [806, 229]},
    'd5':  {'outer_ring_point': [1089, 112], 'inner_ring_point': [1000, 244]},
    'd6':  {'outer_ring_point': [1211, 357], 'inner_ring_point': [1064, 357]},
    'd7':  {'outer_ring_point': [1073, 612], 'inner_ring_point': [995, 472]},
    'd8':  {'outer_ring_point': [849, 614], 'inner_ring_point': [806, 489]},
    'd9':  {'outer_ring_point': [500, 613], 'inner_ring_point': [547, 489]},
    'd10': {'outer_ring_point': [218, 610], 'inner_ring_point': [320, 472]},
}

board_zone = {
    'top_left': [508, 268],
    'top_right': [774, 268],
    'bottom_right': [774, 378],
    'bottom_left': [508, 378],
}

seat_1_definition = {
    'seat_divider_before': 'd2',
    'seat_divider_after': 'd3',
    'label': 'top_center',
    'notes': 'Provisional default for this image. Changeable later if you want a different numbering origin.'
}

def norm(pt):
    return [round(pt[0] / W, 4), round(pt[1] / H, 4)]

instance = {
    'schema_version': '1.0',
    'table_model_id': 'dopo_10max_calibration_instance_current_table_v1_landscape',
    'source_image': str(src),
    'image': {'width': W, 'height': H, 'orientation': 'landscape'},
    'global_geometry': {
        'outer_rail': {
            'kind': 'capsule_from_4_points',
            'points': outer,
            'points_normalized': {k: norm(v) for k, v in outer.items()},
        },
        'inner_rail': {
            'kind': 'capsule_from_4_points',
            'points': inner,
            'points_normalized': {k: norm(v) for k, v in inner.items()},
        },
        'action_area': {
            'kind': 'capsule_from_4_points',
            'points': action,
            'points_normalized': {k: norm(v) for k, v in action.items()},
        },
    },
    'seat_dividers': {
        'count': 10,
        'lines': {
            k: {
                'outer_ring_point': v['outer_ring_point'],
                'inner_ring_point': v['inner_ring_point'],
                'outer_ring_point_normalized': norm(v['outer_ring_point']),
                'inner_ring_point_normalized': norm(v['inner_ring_point']),
            }
            for k, v in seat_dividers.items()
        }
    },
    'board_zone': {
        'kind': 'isosceles_quadrilateral',
        'points': board_zone,
        'points_normalized': {k: norm(v) for k, v in board_zone.items()},
        'logic': {'3_cards': 'flop', '4_cards': 'turn', '5_cards': 'river'}
    },
    'dealer_button_tracking': {
        'allowed_zone': 'boundary_band_around_action_area',
        'position_assignment': 'nearest_seat_wedge_boundary_or_nearest_seat_anchor',
        'notes': [
            'Dealer button may drift slightly into the action area or slightly into the player wedge.',
            'Use nearest assignment while remaining in the allowed boundary band around the action area.'
        ]
    },
    'table_metadata': {
        'seat_1_definition': seat_1_definition,
        'numbering_direction': 'clockwise'
    }
}
json_path.write_text(json.dumps(instance, indent=2), encoding='utf-8')

# overlay
parts = []
parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
parts.append('<defs><style>')
parts.append('.outer{fill:none;stroke:#ff2a2a;stroke-width:7}.inner{fill:none;stroke:#ff5555;stroke-width:5}.action{fill:rgba(255,140,0,0.10);stroke:#ffe066;stroke-width:4}.divider{stroke:#ffe600;stroke-width:4.5;stroke-linecap:round}.board{fill:rgba(37,255,82,0.12);stroke:#25ff52;stroke-width:3}.pt{fill:#ffffff;stroke:#000;stroke-width:1.5}.lbl{font:18px Arial,sans-serif;fill:#fff;font-weight:bold}.seat1{font:20px Arial,sans-serif;fill:#00e5ff;font-weight:bold}')
parts.append('</style></defs>')
parts.append(f'<image href="data:image/jpeg;base64,{img_b64}" x="0" y="0" width="{W}" height="{H}"/>')

def capsule_path(points):
    tl=points['top_left']; tr=points['top_right']; br=points['bottom_right']; bl=points['bottom_left']
    r=(bl[1]-tl[1])/2
    cy=(bl[1]+tl[1])/2
    left_cx=tl[0]
    right_cx=tr[0]
    top_y=tl[1]
    bot_y=bl[1]
    return f'M {left_cx},{top_y} L {right_cx},{top_y} A {r},{r} 0 0 1 {right_cx},{bot_y} L {left_cx},{bot_y} A {r},{r} 0 0 1 {left_cx},{top_y} Z'

parts.append(f'<path class="outer" d="{capsule_path(outer)}"/>')
parts.append(f'<path class="inner" d="{capsule_path(inner)}"/>')
parts.append(f'<path class="action" d="{capsule_path(action)}"/>')

for name, data in seat_dividers.items():
    o = data['outer_ring_point']; i = data['inner_ring_point']
    parts.append(f'<line class="divider" x1="{o[0]}" y1="{o[1]}" x2="{i[0]}" y2="{i[1]}"/>')
    parts.append(f'<circle class="pt" cx="{o[0]}" cy="{o[1]}" r="4"/>')
    parts.append(f'<circle class="pt" cx="{i[0]}" cy="{i[1]}" r="4"/>')
    mx=(o[0]+i[0])/2; my=(o[1]+i[1])/2
    parts.append(f'<text class="lbl" x="{mx+4}" y="{my-4}">{name}</text>')

for group, color in [(outer,'outer'), (inner,'inner'), (action,'action')]:
    pass
for label, pt in outer.items():
    parts.append(f'<circle class="pt" cx="{pt[0]}" cy="{pt[1]}" r="5"/>')
    parts.append(f'<text class="lbl" x="{pt[0]+8}" y="{pt[1]-8}">outer_{label}</text>')
for label, pt in inner.items():
    parts.append(f'<circle class="pt" cx="{pt[0]}" cy="{pt[1]}" r="5"/>')
    parts.append(f'<text class="lbl" x="{pt[0]+8}" y="{pt[1]-8}">inner_{label}</text>')
for label, pt in action.items():
    parts.append(f'<circle class="pt" cx="{pt[0]}" cy="{pt[1]}" r="5"/>')
    parts.append(f'<text class="lbl" x="{pt[0]+8}" y="{pt[1]-8}">action_{label}</text>')

btl = board_zone['top_left']; btr = board_zone['top_right']; bbr = board_zone['bottom_right']; bbl = board_zone['bottom_left']
parts.append(f'<polygon class="board" points="{btl[0]},{btl[1]} {btr[0]},{btr[1]} {bbr[0]},{bbr[1]} {bbl[0]},{bbl[1]}"/>')
for label, pt in board_zone.items():
    parts.append(f'<circle class="pt" cx="{pt[0]}" cy="{pt[1]}" r="5"/>')
    parts.append(f'<text class="lbl" x="{pt[0]+8}" y="{pt[1]-8}">board_{label}</text>')

parts.append('<text class="seat1" x="595" y="170">Seat 1 default = top center wedge (between d2 and d3)</text>')
parts.append('</svg>')
svg_path.write_text('\n'.join(parts), encoding='utf-8')
print(svg_path)
print(json_path)
print(png_path)
