# Independently Recurrent Neural Network 

    ref: Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN (https://arxiv.org/pdf/1803.04831.pdf)

## Indepdently Recurrent Neural Network

The indRNN can be described as

$$
\mathbf{h}_t=\sigma(\mathbf{Wx}_t+\mathbf{u}\odot\mathbf{h}_{t-1}+\mathbf{b})
$$
where recurrent weight $\mathbf{u}$ is a vector and  $\odot$ represents Hadamard product.

Each neuron only receives information from the input and tis own hidden state at the previous time step. The correlation among different neurons can be exploited by stacking two or multiple layers.

### Backpropagation Through Time for An IndRNN

For the $n$-th neuron $h_{n,t}=\sigma(\mathbf{w}_{n}\mathbf{x}_t+u_nh_{n,t-1})$ where the bias is ingored, suppose the objective trying to minimize at time step $T$ is $J_n$. Then the gradient back propagated to the time step $t$ is
$$
\frac{\partial {J_n}}{\partial {h}_{n,t}} =  
\frac{\partial {J_n}}{\partial {h}_{n,T}}\frac{\partial {h}_{n,T}}{\partial {h}_{n,t}}  
=\frac{\partial {J_n}}{\partial {h}_{n,T}} 
\prod_{k=t}^{T-1} \frac{\partial {h}_{n,k+1}}{\partial {h}_{n,k}}  
=\frac{\partial {J_n}}{\partial {h}_{n,T}} \prod_{k=t}^{T-1} {\sigma'}_{n,k+1} {u}_n  
=\frac{\partial {J_n}}{\partial {h}_{n,T}} {u}_n^{T-t} \prod_{k=t}^{T-1} {\sigma'}_{n,k+1} 
$$

It can be seen that the gradient only involves the exponential term of a scalar value $\mu_n$ which can be easily regulated.

### Multiple-layer IndRNN

As mentioned above, neurons in the same IndRNN layer are independent of each other, and cross channel information over time is explored through multiple layers of IndRNN.

Assume a simple $N$-neuron two-layer network where the recurrent weights for the second layer are zero which means the second layer is just a fully connected layer shared onver time. Assume that the activation function is a linear function $\sigma(x) = x$.

Then the first and second layers of a two-layer IndRNN can be represented as following:
$$
\mathbf{h}_{f,t}=\mathbf{W}_f\mathbf{x}_{f,t}+diag(u_{fi})\mathbf{h}_{f,t-1}
$$
$$
\mathbf{h}_{s,t} =\mathbf{W}_s\mathbf{h}_{f,t}
$$

Assuming $\mathbf{W}_s$ is invertible, then 
$$
\mathbf{W}_s^{-1}\mathbf{h}_{s,t}=\mathbf{W}_f\mathbf{x}_{f,t}+diag(u_{fi})\mathbf{W}_s^{-1}\mathbf{h}_{s,t-1}
$$
Thus
$$
\mathbf{h}_{s,t}=\mathbf{W}_s\mathbf{W}_f\mathbf{x}_{f,t}+\mathbf{W}_sdiag(u_{fi})\mathbf{W}_s^{-1}\mathbf{h}_{s,t-1}
$$

By assigning $\mathbf{U}=\mathbf{W}_sdiag(u_{fi})\mathbf{W}_s^{-1}$ and $\mathbf{W}=\mathbf{W}_s\mathbf{W}_f$, it becomes
$$
\mathbf{h}_t=\mathbf{Wx}_t+\mathbf{Uh}_{t-1}
$$
which is a traditional RNN.



