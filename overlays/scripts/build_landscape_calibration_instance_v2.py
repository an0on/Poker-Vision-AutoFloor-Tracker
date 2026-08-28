from pathlib import Path
import base64, json

src = Path('/opt/data/cache/images/img_4f4b3352d112.jpg')
svg_path = Path('/opt/data/tmp/poker_table_calibration_instance_current_table_v2_landscape.svg')
png_path = Path('/opt/data/tmp/poker_table_calibration_instance_current_table_v2_landscape.png')
json_path = Path('/opt/data/tmp/poker_table_calibration_instance_current_table_v2_landscape.json')
img_b64 = base64.b64encode(src.read_bytes()).decode('ascii')

W, H = 1280, 719

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
    'seat_id': 'seat_1',
    'between_dividers': ['d2', 'd3'],
    'label': 'top_center',
    'numbering_direction': 'clockwise',
    'notes': 'Provisional default for this image. Changeable later if desired.'
}

def norm(pt):
    return [round(pt[0] / W, 4), round(pt[1] / H, 4)]

def capsule_path(points):
    tl=points['top_left']; tr=points['top_right']; br=points['bottom_right']; bl=points['bottom_left']
    r=(bl[1]-tl[1])/2
    top_y=tl[1]
    bot_y=bl[1]
    left_cx=tl[0]
    right_cx=tr[0]
    return f'M {left_cx},{top_y} L {right_cx},{top_y} A {r},{r} 0 0 1 {right_cx},{bot_y} L {left_cx},{bot_y} A {r},{r} 0 0 1 {left_cx},{top_y} Z'

def boundary_band(action_points, offset):
    return {
        'top_left': [action_points['top_left'][0]-offset, action_points['top_left'][1]-offset],
        'top_right': [action_points['top_right'][0]+offset, action_points['top_right'][1]-offset],
        'bottom_right': [action_points['bottom_right'][0]+offset, action_points['bottom_right'][1]+offset],
        'bottom_left': [action_points['bottom_left'][0]-offset, action_points['bottom_left'][1]+offset],
    }

button_band_outer = boundary_band(action, 28)
button_band_inner = boundary_band(action, -16)

# Build seat wedges from consecutive divider pairs, wrapping around.
# seat 1 lies between d2 and d3, then clockwise.
seat_order = [
    ('seat_1', 'd2', 'd3'),
    ('seat_2', 'd3', 'd4'),
    ('seat_3', 'd4', 'd5'),
    ('seat_4', 'd5', 'd6'),
    ('seat_5', 'd6', 'd7'),
    ('seat_6', 'd7', 'd8'),
    ('seat_7', 'd8', 'd9'),
    ('seat_8', 'd9', 'd10'),
    ('seat_9', 'd10', 'd1'),
    ('seat_10', 'd1', 'd2'),
]

seat_polygons = {}
seat_anchors = {}
for seat_id, left_div, right_div in seat_order:
    a = seat_dividers[left_div]
    b = seat_dividers[right_div]
    poly = [
        a['outer_ring_point'],
        b['outer_ring_point'],
        b['inner_ring_point'],
        a['inner_ring_point'],
    ]
    seat_polygons[seat_id] = poly
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    seat_anchors[seat_id] = [round(cx, 1), round(cy, 1)]

instance = {
    'schema_version': '1.1',
    'table_model_id': 'dopo_10max_calibration_instance_current_table_v2_landscape',
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
    'seats': {
        seat_id: {
            'divider_before': left_div,
            'divider_after': right_div,
            'seat_wedge_polygon': seat_polygons[seat_id],
            'seat_wedge_polygon_normalized': [norm(p) for p in seat_polygons[seat_id]],
            'seat_anchor': seat_anchors[seat_id],
            'seat_anchor_normalized': norm(seat_anchors[seat_id]),
        }
        for seat_id, left_div, right_div in seat_order
    },
    'board_zone': {
        'kind': 'isosceles_quadrilateral',
        'points': board_zone,
        'points_normalized': {k: norm(v) for k, v in board_zone.items()},
        'logic': {'3_cards': 'flop', '4_cards': 'turn', '5_cards': 'river'}
    },
    'dealer_button_tracking': {
        'allowed_zone': 'boundary_band_around_action_area',
        'boundary_band_px': {
            'outer_band_offset_from_action_area': 28,
            'inner_band_offset_from_action_area': 16
        },
        'boundary_band_outer': button_band_outer,
        'boundary_band_outer_normalized': {k: norm(v) for k, v in button_band_outer.items()},
        'boundary_band_inner': button_band_inner,
        'boundary_band_inner_normalized': {k: norm(v) for k, v in button_band_inner.items()},
        'position_assignment': 'nearest_seat_wedge_boundary_or_nearest_seat_anchor',
        'notes': [
            'Dealer button may drift slightly into the action area or slightly into the player wedge.',
            'Use nearest assignment while remaining in the allowed boundary band around the action area.'
        ]
    },
    'table_metadata': {
        'seat_1_definition': seat_1_definition
    },
    'runtime_rules': {
        'seat_occupancy_priority': ['chips', 'all_in_button', 'face_down_cards_secondary'],
        'board_state_source': 'card_count_in_board_zone'
    }
}
json_path.write_text(json.dumps(instance, indent=2), encoding='utf-8')

parts = []
parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
parts.append('<defs><style>')
parts.append('.outer{fill:none;stroke:#ff2a2a;stroke-width:7}.inner{fill:none;stroke:#ff5555;stroke-width:5}.action{fill:rgba(255,140,0,0.10);stroke:#ffe066;stroke-width:4}.bando{fill:none;stroke:#00d4ff;stroke-width:3;stroke-dasharray:10 8}.bandi{fill:none;stroke:#00d4ff;stroke-width:2;stroke-dasharray:6 6}.divider{stroke:#ffe600;stroke-width:4.5;stroke-linecap:round}.board{fill:rgba(37,255,82,0.12);stroke:#25ff52;stroke-width:3}.pt{fill:#ffffff;stroke:#000;stroke-width:1.5}.lbl{font:17px Arial,sans-serif;fill:#fff;font-weight:bold}.seat{fill:rgba(255,230,0,0.10);stroke:#ffd000;stroke-width:1.4}.seat1{font:20px Arial,sans-serif;fill:#00e5ff;font-weight:bold}')
parts.append('</style></defs>')
parts.append(f'<image href="data:image/jpeg;base64,{img_b64}" x="0" y="0" width="{W}" height="{H}"/>')
parts.append(f'<path class="outer" d="{capsule_path(outer)}"/>')
parts.append(f'<path class="inner" d="{capsule_path(inner)}"/>')
parts.append(f'<path class="action" d="{capsule_path(action)}"/>')
parts.append(f'<path class="bando" d="{capsule_path(button_band_outer)}"/>')
parts.append(f'<path class="bandi" d="{capsule_path(button_band_inner)}"/>')
for seat_id, poly in seat_polygons.items():
    pts = ' '.join(f'{p[0]},{p[1]}' for p in poly)
    parts.append(f'<polygon class="seat" points="{pts}"/>')
    a = seat_anchors[seat_id]
    parts.append(f'<text class="lbl" x="{a[0]-18}" y="{a[1]}">{seat_id}</text>')
for name, data in seat_dividers.items():
    o = data['outer_ring_point']; i = data['inner_ring_point']
    parts.append(f'<line class="divider" x1="{o[0]}" y1="{o[1]}" x2="{i[0]}" y2="{i[1]}"/>')
    parts.append(f'<circle class="pt" cx="{o[0]}" cy="{o[1]}" r="4"/>')
    parts.append(f'<circle class="pt" cx="{i[0]}" cy="{i[1]}" r="4"/>')
for label, pt in {**{f'outer_{k}':v for k,v in outer.items()}, **{f'inner_{k}':v for k,v in inner.items()}, **{f'action_{k}':v for k,v in action.items()}}.items():
    parts.append(f'<circle class="pt" cx="{pt[0]}" cy="{pt[1]}" r="5"/>')
    parts.append(f'<text class="lbl" x="{pt[0]+6}" y="{pt[1]-6}">{label}</text>')
for label, pt in board_zone.items():
    parts.append(f'<circle class="pt" cx="{pt[0]}" cy="{pt[1]}" r="5"/>')
    parts.append(f'<text class="lbl" x="{pt[0]+6}" y="{pt[1]-6}">board_{label}</text>')
pts = ' '.join(f'{board_zone[k][0]},{board_zone[k][1]}' for k in ['top_left','top_right','bottom_right','bottom_left'])
parts.append(f'<polygon class="board" points="{pts}"/>')
parts.append('<text class="seat1" x="582" y="170">Seat 1 default = top center wedge</text>')
parts.append('<text class="lbl" x="930" y="210">cyan dashed = dealer button allowed band</text>')
parts.append('</svg>')
svg_path.write_text('\n'.join(parts), encoding='utf-8')
print(svg_path)
print(json_path)
print(png_path)
