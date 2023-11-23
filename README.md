# GPTCN
The source code for GPTCN is located in the source_code folder.

In the following, some additional remarks are made on areas not mentioned in detail in the icassp article. It includes the topology diagram of each module of the model and the hyperparameter experiment after grid search.

### The model topology of the GPTCN
![image text](https://github.com/Moloyiker/GPTCN/blob/main/Static_file/picture/The%20model%20topology%20of%20the%20GPTCN.png "The model topology of the GPTCN")

### An illustration of user installation behavior
![image text](https://github.com/Moloyiker/GPTCN/blob/main/Static_file/picture/An%20illustration%20of%20user%20installation%20behavior.png "An illustration of user installation behavior")
In actual application installation behavior data, as we can observe from the Figure above, the occurrence of user behavior is often low-frequency and unevenly distributed over time. After the massive demand for downloads associated with replacing a new phone is largely satisfied, users tend to download and install apps only when necessary. Additionally, due to external influences in real life, users may be impulsive and install or uninstall multiple applications rapidly within a short period of time. The resulting time intervals of such behavior are often non-negligible. The sparsity and unevenness of behavior data can often have a negative impact on the performance of models.

### Diagram of the structure of our data scheme
![image](https://github.com/Moloyiker/GPTCN/assets/63031589/846f90eb-593b-4cf9-90ab-729d6350067f)

### TCN autoencoder architecture diagram
![image](https://github.com/Moloyiker/GPTCN/assets/63031589/24b2f2cd-77cf-4b8b-bf57-b52cdeaf5132)

### Gated multi-head attention Architecture
![image](https://github.com/Moloyiker/GPTCN/assets/63031589/2105fe5a-5538-44f5-8e4c-df791f69c273)


### Performance of DDHCN with various hyperparameters on age and gender prediction
![Performance of DDHCN with various hyperparameters on age and gender prediction](https://github.com/Moloyiker/GPTCN/assets/63031589/444f25c9-2f31-42fb-a85d-4d941edfefe6)

Our model is an improved version of the basic model consisting of Transformer and CNN.  The number of layers in the Transformer and the size of the convolutional kernel are hyperparameters known to have an impact on the final performance of the model.  Here, we perform a grid search on the number of layers in the Transformer encoder and decoder, ranging from 1 to 5, as shown in Figure (a) and (b). The width of the convolutional kernel is selected as 2, 4, 8, 16, 32 for testing, as shown in Figure (c). After experimentation, we recommend using two layers of Transformer and setting the width of the convolutional kernel to 8 for optimal performance.
