from pprint import pprint

with open("classified.txt") as f:
    lines = f.readlines()

values = []
for line in lines:
    target, classes = line.split(":")
    values.append((target, classes.count(",") - classes.count(target)))


# pprint(values)

pprint(sorted(values, key=lambda o: o[1], reverse=True )[:5]   )
