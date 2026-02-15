import keyboard_module as kbm


class ActionMapping:

    _SIMPLE_ACTIONS = {
        "downArrow": kbm.pressDownArrow,
        "upArrow": kbm.pressUpArrow,
        "space": kbm.pressSpace,
    }

    _TAB_NAV_ACTIONS = {
        "nextTab": kbm.pressRightArrow,
        "prevTab": kbm.pressLeftArrow,
        "upTab": kbm.pressUpArrow,
        "downTab": kbm.pressDownArrow,
    }

    def __init__(self, action_labels):
        self.altTabIsPressed = False
        self.count = 0
        self.action_labels = action_labels
        print("labels:", self.action_labels)

    def reset(self):
        self.count += 1
        if self.count > 10:
            self.altTabIsPressed = False
            kbm.closeAltTab()
            self.count = 0

    def convert_signs_to_array(self, signs):
        """
        builds an array of left and right hands signs
        :param signs: an array of signs
        :return: an array of hands
        """
        hands = []
        for sign in signs:
            if sign[2] == "Left":
                hands.append(self.action_labels[sign[0]])
        if len(hands) == 0:
            hands.append("None")

        for sign in signs:
            if sign[2] == "Right":
                hands.append(self.action_labels[sign[0]])
        if len(hands) == 1:
            hands.append("None")

        return hands

    def execute_action(self, action):
        if action == "openAltTab" and not self.altTabIsPressed:
            self.altTabIsPressed = True
            kbm.openAltTab()
            return

        if action == "closeAltTab" and self.altTabIsPressed:
            self.altTabIsPressed = False
            kbm.closeAltTab()
            return

        if self.altTabIsPressed and action in self._TAB_NAV_ACTIONS:
            self._TAB_NAV_ACTIONS[action]()
            return

        if action in self._SIMPLE_ACTIONS:
            self._SIMPLE_ACTIONS[action]()

