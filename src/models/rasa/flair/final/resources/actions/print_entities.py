from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.actions.action import Action
import json

class ActionPrintEntities(Action):
    def name(self):
        return 'action_print_entities'

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message('test')
        dispatcher.utter_message(json.dumps(tracker.latest_message))
        return