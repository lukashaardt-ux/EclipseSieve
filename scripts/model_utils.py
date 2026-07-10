from xgboost import XGBClassifier

class BalancedXGBClassifier(XGBClassifier):
    '''
    Recomputes scale_pos_weight from whatever y it's actually given at
    fit-time, so it behaves like class_weight='balanced' even when cloned
    and refit on a different fold/subset than the one it was first built with.
    '''
    def fit(self, X, y, **kwargs):
        neg, pos = (y == 0).sum(), (y == 1).sum()
        self.scale_pos_weight = neg / pos
        return super().fit(X, y, **kwargs)