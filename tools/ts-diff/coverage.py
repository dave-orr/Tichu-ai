"""Counts rule edge cases exercised by a trace file (from `tichu trace`)."""
import collections, json, sys

c = collections.Counter()
for line in open(sys.argv[1]):
    t = json.loads(line)
    c['rounds'] += 1
    for st in t['steps']:
        a, b, ch = st['action'], st['before'], st['checks']
        c['action:' + a['type']] += 1
        cards = a.get('cards') or []
        if a['type'] == 'wish' and a['rank']:
            c['wish named'] += 1
        if ch['turn'] and b['trick'] and b['wish'] and not ch['turn']['canPass']:
            c['pass forbidden by wish'] += 1
        if ch['turn'] and b['wish'] and not b['trick']:
            c['lead while wish open'] += 1
        if cards == [{'type': 'special', 'name': 'dog'}]:
            c['dog'] += 1
            if b['outOrder'][(a['seat'] + 2) % 4]:
                c['dog, partner out'] += 1
        if any(x.get('name') == 'phoenix' for x in cards) and len(cards) >= 5:
            c['phoenix in 5+ combo'] += 1
        if a['type'] == 'bomb' and b['countdown'] is not None:
            c['bomb during countdown'] += 1
        if a['type'] == 'giveDragon' and sum(1 for o in b['outOrder'] if o) >= 3:
            c['dragon giveaway ending round'] += 1
    o = t['final']['outOrder']
    first = [s for s in range(4) if o[s] == 1]
    second = [s for s in range(4) if o[s] == 2]
    if first and second and first[0] % 2 == second[0] % 2:
        c['1-2 finish'] += 1
for k, v in sorted(c.items()):
    print(f'{k:32} {v}')
