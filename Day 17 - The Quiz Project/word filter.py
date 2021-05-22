class WordFilter:

    def __init__(self, words):
        self.words = words

    def f(self, prefix: str, suffix: str) -> int:
        for word in self.words:
            if word.lower().startswith(prefix.lower()) and word.lower().endswith(suffix.lower()):
                return self.words.index(word)


# Your WordFilter object will be instantiated and called as such:
obj = WordFilter(['first', 'second', 'third', 'The last one'])
param_1 = obj.f('the', 'one')
print(param_1)