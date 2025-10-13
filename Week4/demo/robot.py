from environment import Environment


class Robot():

    def __init__(self, position: int):
        self.position = position

    def sense(self, environment):
        return environment.get_cell(self.position)

    def decide(self, percept):
        if percept == "d":
            return "clean"
        else:
            return "move"

    def act(self, environment):
        cell = self.sense(environment)
        decision = self.decide(cell)

        if decision == "clean":
            environment.clear_cell(self.position)
        elif decision == "move":
            to = 1
            if self.position == 1:
                to = 0
            self.move(environment, to)

    def move(self, environment, to):
        if environment.move_robot(to):
            self.position = to
