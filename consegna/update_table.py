import numpy as np

# You have the following variables at disposal:
# @. s        : current state, next state
# @. a        : selected action, given s
# @. s2, r    : the next state and the gained reward returned by the environment, given s and a
#
# @. TABLE    : the table Q learned until now,
# @. gamma    : Q-learning parameter,
# @. alpha    : the constant learning rate. RECALL: you set it by command line!
#
def student_update_rule(s,s2,a,r,TABLE,gamma,alpha):
    # TODO: INSERT YOUR CODE HERE
    #Tip: TABLE[s+(a,)] is the entry of the table relative to state s and actison a
    TABLE[s+(a,)] = (1-alpha)*TABLE[s+(a,)] + alpha*(r + gamma*np.max(TABLE[s2]))
    return TABLE;