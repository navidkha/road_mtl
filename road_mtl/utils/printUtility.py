
from colorama import init, Fore
import random


init(convert=True)


def print_success(msg):
    print(Fore.GREEN + msg + Fore.RESET)


def print_info(msg):
    print(Fore.BLUE + msg + Fore.RESET)


def print_warn(msg):
    print(Fore.YELLOW + msg + Fore.RESET)


def print_error(msg):
    print(Fore.RED + msg + Fore.RESET)


def print_cyan(msg):
    print(Fore.CYAN + msg + Fore.RESET)


def print_magenta(msg):
    print(Fore.MAGENTA + msg + Fore.RESET)

def random_color():
    rgb = [255, 0, 0]
    random.shuffle(rgb)
    return tuple(rgb)
