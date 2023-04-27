import pandas as pd
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv('training.csv')

# Define the predictor variables
predictors = data.drop(['ONCOGENIC'], axis=1)

# Define the target variable
target = data['ONCOGENIC']

# Define a function to perform stepwise regression
def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            if verbose:
                print(f'Add {best_feature} with p-value {best_pval}')
            changed = True
        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f'Drop {worst_feature} with p-value {worst_pval}')
        if not changed:
            break
    return included

# Perform stepwise regression
selected_vars = stepwise_selection(predictors, target)

# Print the selected variables
print(selected_vars)