names = {}


def main(name):
    if name not in names.keys():
        names[name] = 1
        return 'OK'
    else:
        return gen_name(name)


def gen_name(old_name):
    new_name = f"{old_name}{names[old_name]}"
    names[old_name] += 1
    if new_name in names.keys():
        names[new_name] += 1
    else:
        names[new_name] = 1
    return new_name


if __name__ == '__main__':
    n = int(input())
    for _ in range(n):
        name = input()
        print(main(name))
