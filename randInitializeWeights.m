function W = randInitializeWeights(L_in, L_out)
% W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
% of a layer with L_in incoming connections and L_out outgoing connections.
% Random inialization instead of zero initialization is required for
% breaking the symmetry while training the neural network. 

W = rand(L_out, 1 + L_in); %  1 + L_in because the first column of W handles the "bias" terms

init_epsilon = 0.01;
W = 2*init_epsilon * W - init_epsilon;

end 