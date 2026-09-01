# Poker table calibration format v1

## Goal
A calibration format that matches your intended workflow:
- outer rail with 4 points
- inner oval with 4 points
- action area with 4 points
- each seat divider with 2 points
- one single board zone with 4 points

## 1. Capsule / oval fitting with 4 points
For these three shapes:
- outer rail
- inner rail
- action area

we use the same construction.

### Required points
Pick 4 points in clockwise order:
- `top_left`
- `top_right`
- `bottom_right`
- `bottom_left`

These are **not arbitrary corner points**.
They should lie on the **long straight parts** of the capsule-like oval.

### Why this works
With those 4 points we can derive:
- the upper straight line
- the lower straight line
- the common radius
- the left semicircle center
- the right semicircle center

So the shape becomes a proper **capsule / racetrack oval** instead of a rough freehand polygon.

### Recommended placement
For each of the three shapes:
- `top_left` and `top_right`: on the upper straight segment
- `bottom_left` and `bottom_right`: on the lower straight segment
- keep them horizontally aligned as closely as possible
- do not place them into the rounded ends

## 2. Seat dividers with 2 points each
Each seat divider is one line.

### Required points per divider
- `outer_ring_point`
- `inner_ring_point`

Meaning:
- one point on the outer seating ring
- one point on the inner/action boundary

This creates one clean radial separator.

### Result
10 divider lines produce the 10 seat wedges.

## 3. Board zone with 4 points
The board is one single zone.

### Required points
- `top_left`
- `top_right`
- `bottom_right`
- `bottom_left`

This should form a clean, symmetric, almost isosceles quadrilateral over the full board placement area.

### Runtime logic
Inside this one zone:
- 3 visible cards = flop
- 4 visible cards = turn
- 5 visible cards = river

This means you do **not** need separate flop/turn/river boxes for state progression.

## 4. Detection rules from this schema
### Dealer button path
A dedicated dealer-button band is **not required**.

Reason:
- the dealer button is visually distinctive
- it can be detected directly
- then assigned to the nearest player area / seat wedge

### Dealer button assignment rule
Use **nearest** assignment:
- detect the dealer button directly
- assign it to the nearest valid player area / seat anchor

This is the simpler and better practical rule if the dealer button is reliably detectable.

### Seat occupancy priority
1. chips
2. all-in button
3. face-down cards as secondary confirmation

### Community board state
Only count cards in the board zone.

## 5. Seat 1 must be fixed at calibration end
At the end of calibration, one derived seat wedge must be explicitly marked as **Seat 1**.

Then:
- the remaining seats are numbered clockwise
- dealer-button movement can be mapped to real seat numbers
- downstream Tournament Director / seat-mapping logic stays deterministic

## 6. Practical calibration order
1. rotate image into canonical orientation
2. set outer rail 4 points
3. set inner rail 4 points
4. set action area 4 points
5. set 10 divider lines with 2 points each
6. set board zone 4 points
7. derive seat wedges automatically
8. explicitly assign Seat 1
9. save calibration JSON

## 7. Files
Machine-readable schema:
`/opt/data/tmp/poker_table_calibration_schema_v1.json`
