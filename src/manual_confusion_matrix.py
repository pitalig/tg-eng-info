#%% [markdown]

# |                      | (B) Face (previsto) | (xB) Não Face (previsto) |      |
# |----------------------|---------------------|--------------------------|------|
# | (A) Face (real)      | B_A                 | xB_A                     | P_A  |
# | (xA) Não Face (real) | B_xA                | xB_xA                    | P_xA |
# |                      | P_B                 | P_xB                     | 1    |

#%% Calculate percentage of images in each group

results_total = result_faces["count_imgs"] + result_no_faces["count_imgs"] # Universo
P_A = result_faces["count_imgs"] / results_total
P_xA = result_no_faces["count_imgs"] / results_total
P_B = (result_no_faces["count_1+"] + result_faces["count_1+"]) / results_total
P_xB = (result_no_faces["count_0"] + result_faces["count_0"]) / results_total
print([P_A, P_xA, sum([P_A, P_xA])])
assert sum([P_A, P_xA]) == 1, "Todas imagens devem ter ou nao faces"
print([P_B, P_xB, sum([P_B, P_xB])])
assert sum([P_B, P_xB]) == 1, "Todas previsoes devem ter ou nao faces"

#%% Calculate intersection between groups

P_A_n_B = result_faces["count_1+"] / results_total
P_xA_n_B = result_no_faces["count_1+"] / results_total
P_A_n_xB = result_faces["count_0"] / results_total
P_xA_n_xB = result_no_faces["count_0"] / results_total
print([P_A_n_B, P_xA_n_B, P_A_n_xB, P_xA_n_xB, sum([P_A_n_B, P_xA_n_B, P_A_n_xB, P_xA_n_xB])])
assert sum([P_A_n_B, P_xA_n_B, P_A_n_xB, P_xA_n_xB]) == test_check, "Todas intersecoes somam 1"

#%% Generate confusion matrix and calculate sensitivity and specificity

P_B_given_A = P_A_n_B / P_A
P_xB_given_A = P_A_n_xB / P_A
P_B_given_xA = P_xA_n_B / P_xA
P_xB_given_xA = P_xA_n_xB / P_xA
print([P_B_given_A, P_xB_given_A, sum([P_B_given_A, P_xB_given_A])])
assert sum([P_B_given_A, P_xB_given_A]) == 1, "Uma nao face pode so ser prevista como face ou nao"
print([P_B_given_xA, P_xB_given_xA, sum([P_B_given_xA, P_xB_given_xA])])
assert sum([P_B_given_xA, P_xB_given_xA]) == 1, "Uma face pode so ser prevista como face ou nao"

print("sensitividade")
print(P_B_given_A)
print("especificidade")
print(P_xB_given_xA)

result_print = [[P_A_n_B*100, P_A_n_xB*100, P_A*100],
                [P_xA_n_B*100, P_xA_n_xB*100, P_xA*100],
                [P_B*100, P_xB*100, 100]]
pd.DataFrame(result_print, columns=["(B) Face (previsto)", "(xB) Não Face (previsto)", ""], index=["(A) Face (real)", "(xA) Não Face (real)", ""])
