import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold

import asl_utils


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("inf")
        best_model = None

        for hidden_states_number in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=hidden_states_number, n_iter=1000).fit(self.X, self.lengths)
                log_likelihood = model.score(self.X, self.lengths)
                data_points_number = self.X.shape[0]
                features_number = self.X.shape[1]
                parameters_number = hidden_states_number * features_number * 2 + \
                                    hidden_states_number * hidden_states_number - 1
                bic_score = -2 * log_likelihood + parameters_number * np.log(data_points_number)
                if bic_score < best_score:
                    best_score = bic_score
                    best_model = model

            except Exception as e:
                continue

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("-inf")
        best_model = None
        number_of_words = len(self.hwords.keys())

        for hidden_states_number in range(self.min_n_components, self.max_n_components):
            sum_log_likelihood_other_words = 0
            try:
                model = GaussianHMM(n_components=hidden_states_number, n_iter=1000).fit(self.X, self.lengths)
                sum_log_likelihood_matching_words = model.score(self.X, self.lengths)

                for word in self.hwords.keys():
                    other_word_data_points, lengths = self.hwords[word]
                    log_likelihood_other_words = model.score(other_word_data_points, lengths)
                    sum_log_likelihood_other_words += log_likelihood_other_words

            except Exception as e:
                break

            dic_score = sum_log_likelihood_matching_words - (1 / (number_of_words - 1)) * (
                sum_log_likelihood_other_words - sum_log_likelihood_matching_words)

            if dic_score > best_score:
                best_score = dic_score
                best_model = model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        k_fold = KFold()
        best_score = float("-inf")
        best_model = None

        for hidden_states_number in range(self.min_n_components, self.max_n_components + 1):
            scores = []
            model = None

            if len(self.sequences) < 3:
                break

            for training_set_indices, test_set_indices in k_fold.split(self.sequences):
                training_set, training_set_lengths = asl_utils.combine_sequences(training_set_indices, self.sequences)
                test_set, test_set_lengths = asl_utils.combine_sequences(test_set_indices, self.sequences)

                try:
                    model = GaussianHMM(n_components=hidden_states_number, n_iter=1000).fit(training_set,
                                                                                            training_set_lengths)
                    log_likelihood = model.score(test_set, test_set_lengths)
                    scores.append(log_likelihood)
                except Exception as e:
                    break

            average_score = np.average(scores)
            if average_score > best_score:
                best_score = average_score
                best_model = model

        return best_model if best_model is not None else self.base_model(self.n_constant)
