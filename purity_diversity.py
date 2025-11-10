class PurityDiversity:     #low for pure, high for diverse
    def __init__(self, balance=None):
        self.balance = balance
        self.score_bounds={}
        
    def update(self, sample):
        loss = sample['loss']
        uncertainty = sample['uncertainty']
        concept = sample['concept']
        score = (1- self.balance) * loss + self.balance * uncertainty

        #update minimum and maximum scores
        if concept not in self.score_bounds:
            self.score_bounds[concept] = {'min':score,'max':score}
        else:
            self.score_bounds[concept]['min'] = min(self.score_bounds[concept]['min'], score)
            self.score_bounds[concept]['max'] = max(self.score_bounds[concept]['max'], score)
        
        # Normalize the score
        bounds = self.score_bounds[concept]
        norm_score = (score - bounds['min']) / (bounds['max'] - bounds['min'] + 1e-8)

        return norm_score