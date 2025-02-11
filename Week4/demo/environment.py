import utils


class Environment:

    def __init__(self):
        self.world = ['d','d']
        self.robot_location = 0

    def get_cell(self, position: int):
        return self.world[position]

    def clear_cell(self, position: int):
        self.world[position] = "-"
        return True

    def move_robot(self, move_to):
        self.robot_location = move_to
        return True

    def __str__(self):
        out = "\t".join(self.world) + "\n"
        if self.robot_location == 1:
            out += "\t"
        return out + "r"


if __name__ == "__main__":
    e = Environment()
    r = utils.Robot(0)
    print(e)

    for i in range(10):
        r.act(e)
        print()
        print(e)
        input()


