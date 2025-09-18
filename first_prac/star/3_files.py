os = {}
modes = {'read': 'r',
         'write': 'w',
         'execute': 'x', }


def gen_files(string):
    data = string.split()
    name = data.pop(0)
    mode = data
    os[name] = mode
    return os


def try_use(string):
    op, file = string.split()
    if modes[op] in os[file]:
        print('OK')
    else:
        print('Access denied')

def main():
    for _ in range(int(input())):
        gen_files(input())
    for _ in range(int(input())):
        try_use(input())

if __name__ == '__main__':
    main()
