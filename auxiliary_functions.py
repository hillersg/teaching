import numpy as np
import matplotlib.pyplot as plt

def aux_plot_kcosinesine(k,n,data,ktestcos,ktestsin):
    '''
    plot cosine and sine test functions 
    '''
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,4))

    axes[0].plot(n,data,'.--',color='gray',label='data')
    axes[0].plot(n,ktestcos,'c.-',label=('cos(%i 2$\pi$n/N)'%(k)))
    axes[0].set_xlabel('n')
    axes[0].set_yticks([-1, 0, 1])
    axes[0].set_title('Cosine test function')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(bbox_to_anchor=(1.05, 0.95), loc='upper right', borderaxespad=0.)

    axes[1].plot(n,data,'.--',color='gray',label='data')
    axes[1].plot(n,ktestsin,'m.-',label=('sin(%i 2$\pi$n/N)'%(k)))
    axes[1].set_xlabel('n')
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_title('Sine test function')
    axes[1].legend(bbox_to_anchor=(1.05, 0.95), loc='upper right', borderaxespad=0.)

    plt.show()

    return

def aux_plot_modeldecomposition(n,data,newdata,dataargument,decompose):
    '''
    plotting the model decomposition
    '''
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,4))

    axes[0].plot(n,data,'--',color='gray',label='original data');  
    axes[0].plot(n,newdata,'r-',label='phase-shifted data');       
    axes[0].set_yticks([-1, 0, 1]);                     
    axes[0].set_xlabel('n');                       
    axes[0].set_title('Data');                
    axes[0].set_ylabel('Amplitude');
    axes[0].legend(bbox_to_anchor=(1.05, 0.95), loc='upper right', borderaxespad=0.);

    alpha = []
    beta  = []

    if decompose: 
        # never mind some magic ...
        # build model vectors based on sine and cosine functions
        model_sin = np.sin(dataargument);
        model_cos = np.cos(dataargument);
        GT = np.concatenate(([model_sin], [model_cos]));
        G = GT.T # shape (N,2)
        alpha, beta = np.linalg.lstsq(np.matmul(GT,G),np.matmul(GT,newdata),rcond=None)[0];
        # alpha and beta are the weights of the sine and cosine basis function
        #print("The sine- and cosine-weights are %5.3f, %5.3f"%(alpha, beta));
        model = alpha * model_sin + beta * model_cos;

        axes[0].plot(n,model,'k--',linewidth=2,dashes=(5, 3),label='model');                     
        axes[0].set_title('Data and model');                
        axes[0].legend(bbox_to_anchor=(1.05, 0.95), loc='upper right', borderaxespad=0.);

        axes[1].plot(n,alpha*model_sin,'m',label='sine-part');
        axes[1].plot(n,beta*model_cos,'c',label='cosine-part');
        axes[1].plot(n,model,'k--',linewidth=2,dashes=(5, 3),label='model');
        axes[1].set_yticks([-1, 0, 1]);
        axes[1].set_xlabel('n');
        axes[1].set_title('Model decomposition');
        axes[1].legend(bbox_to_anchor=(1.05, 0.95), loc='upper right', borderaxespad=0.);

    plt.show()

    return alpha, beta
