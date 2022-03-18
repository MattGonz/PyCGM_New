from model import Model


class PyCGM():
    def __init__(self, subjects):
        if isinstance(subjects, Model):
            subjects = [subjects]

        self.subjects = subjects


    def run_all(self):
        for i, subject in enumerate(self.subjects):
            print(f"Running subject {i+1} of {len(self.subjects)}")
            subject.run()


    def __getitem__(self, index):
        return self.subjects[index]

