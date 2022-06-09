import shell
import util
import wordsegUtil

############################################################
# Problem 1: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def start(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.query
        # END_YOUR_CODE

    def goalp(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return len(state) == 0
        # END_YOUR_CODE

    def expand(self, state):
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        results = []
        for i in range(len(state), 0, -1):
            action = state[:i]
            remaining = state[len(action):]
            cost = self.unigramCost(action)
            results.append((action, remaining, cost))
        return results
        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    words = ' '.join(ucs.actions)
    return words
    # END_YOUR_CODE

############################################################
# Problem 2: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def start(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_CODE

    def goalp(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return state[0] == len(self.queryWords)
        # END_YOUR_CODE

    def expand(self, state):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        results = []
        # state[0] = current index a set of strings {'a', 'b',...}
        # state[1] = previous words
        actions = self.possibleFills(self.queryWords[state[0]])
        if len(actions) == 0:
            actions.add(self.queryWords[state[0]])
        #for i in range(state[0], len(self.queryWords)):
            #actions = self.possibleFills(self.queryWords[i])
        for action in actions:
            cost = self.bigramCost(state[1], action)
            results.append((action, (state[0]+1, action), cost))
        return results
        # END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    if len(queryWords) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    words_vowels = ' '.join(ucs.actions)
    return words_vowels
    # END_YOUR_CODE


if __name__ == '__main__':
    shell.main()
