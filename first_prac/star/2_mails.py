names = []


def main(name):
    if name not in names:
        names.append(name)
        return 'OK'
    else:
        return gen_name(name)


def gen_name(old_name):
    new_name = f"{old_name}{names.count(old_name)}"
    return new_name


if __name__ == '__main__':
    n = int(input())
    for _ in range(n):
        name = input()
        print(main(name))
