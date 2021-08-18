import os

import requests

class DbApiUtils:
    def __init__(self, base_url=None):
        if base_url is None:
            self.base_url = "http://bigkahuna.corp.alleninstitute.org/api"
        else:
            self.base_url = base_url
    
    def get_specimens(self):
        specimens_url = os.path.join(self.base_url, "specimens")

        r = requests.get(specimens_url)
        specimens = r.json()

        return specimens

    def get_specimen(self, specimen_id):
        specimen_url = os.path.join(self.base_url, specimen_id)

        r = requests.get(specimen_url)
        specimen = r.json()

        return specimen
    
    def get_section(self, specimen_id, section_num):
        section_url = os.path.join(self.base_url, specimen_id, section_num)
        print(section_url)

        r = requests.get(section_url)
        section = r.json()

        return section

    def add_session(self, session):
        session_url = os.path.join(self.base_url,
                                   session["specimen_id"],
                                   "new_session")
        print(session_url)

        r = requests.post(session_url, json=session)
        print(r.status_code)
        result = r.json()

        return result

    def get_session(self, specimen_id, session_id):
        session_url = os.path.join(self.base_url,
                                   specimen_id,
                                   "session",
                                   session_id)

        r = requests.get(session_url)
        session = r.json()

        return session

    def add_acquisition(self, acquisition):
        acquisition_url = os.path.join(self.base_url)

        r = requests.post(acquisition_url, acquisition)
        result = r.json()

        return result

    def get_acquisition(self, acquisition_id):
        acquisition_url = os.path.join(self.base_url)

        r = requests.get(acquisition_url)
        acquisition = r.json()

        return acquisition