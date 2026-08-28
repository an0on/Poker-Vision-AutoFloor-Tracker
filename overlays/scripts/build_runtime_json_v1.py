from pathlib import Path
import json

src = Path('/opt/data/tmp/poker_table_calibration_instance_current_table_v3_landscape.json')
out = Path('/opt/data/tmp/poker_table_runtime_v1.json')

cal = json.loads(src.read_text(encoding='utf-8'))

W = cal['image']['width']
H = cal['image']['height']

def norm(pt):
    return [round(pt[0] / W, 4), round(pt[1] / H, 4)]

def bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return {
        'x': round(x1, 1),
        'y': round(y1, 1),
        'w': round(x2 - x1, 1),
        'h': round(y2 - y1, 1),
    }

def inset_rect(rect, dx, dy):
    return {
        'x': round(rect['x'] + dx, 1),
        'y': round(rect['y'] + dy, 1),
        'w': round(max(1, rect['w'] - 2 * dx), 1),
        'h': round(max(1, rect['h'] - 2 * dy), 1),
    }

def rect_norm(rect):
    return {
        'x': round(rect['x'] / W, 4),
        'y': round(rect['y'] / H, 4),
        'w': round(rect['w'] / W, 4),
        'h': round(rect['h'] / H, 4),
    }

runtime = {
    'schema_version': '1.0',
    'runtime_model_id': 'dopo_10max_runtime_v1',
    'based_on_calibration': str(src),
    'image': cal['image'],
    'table': {
        'outer_rail': cal['global_geometry']['outer_rail'],
        'inner_rail': cal['global_geometry']['inner_rail'],
        'action_area': cal['global_geometry']['action_area'],
        'board_zone': cal['board_zone'],
    },
    'seat_1_definition': cal['table_metadata']['seat_1_definition'],
    'numbering_direction': cal['table_metadata']['seat_1_definition']['numbering_direction'],
    'dealer_button_tracking': {
        'detector_required': True,
        'position_assignment': 'nearest_player_area_or_nearest_seat_anchor',
        'fallback': 'nearest_seat_wedge_polygon',
    },
    'board_state_logic': {
        'zone': 'board_zone',
        '3_cards': 'flop',
        '4_cards': 'turn',
        '5_cards': 'river',
    },
    'seat_runtime': {}
}

for seat_id, seat in cal['seats'].items():
    poly = seat['seat_wedge_polygon']
    box = bbox(poly)
    player_area = box
    chip_zone = inset_rect(box, box['w'] * 0.16, box['h'] * 0.18)
    card_zone = inset_rect(box, box['w'] * 0.27, box['h'] * 0.30)
    runtime['seat_runtime'][seat_id] = {
        'seat_id': seat_id,
        'divider_before': seat['divider_before'],
        'divider_after': seat['divider_after'],
        'seat_anchor': seat['seat_anchor'],
        'seat_anchor_normalized': norm(seat['seat_anchor']),
        'seat_wedge_polygon': poly,
        'seat_wedge_polygon_normalized': [norm(p) for p in poly],
        'player_area': player_area,
        'player_area_normalized': rect_norm(player_area),
        'chip_zone': chip_zone,
        'chip_zone_normalized': rect_norm(chip_zone),
        'card_presence_zone': card_zone,
        'card_presence_zone_normalized': rect_norm(card_zone),
        'occupancy_priority': ['chips', 'all_in_button', 'face_down_cards_secondary'],
    }

out.write_text(json.dumps(runtime, indent=2), encoding='utf-8')
print(out)
