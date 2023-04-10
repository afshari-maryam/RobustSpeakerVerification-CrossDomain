# RobustSpeakerVerification-CrossDomain
<br>Robustness of speaker verification systems against domain mismatch</br>
  1. 
      Description: <br/>
      
      In this idea, we use a DANN architecture and instead of its coder model, a deep learning model such as x-vector is placed, whose input is supposed to be the feature extracted from large pre-trained models such as wav2vec2 instead of extracted features such as MFCC. The representations of all hidden layers of the pre-trained model are first averaged with learnable weights and then fed into the encoder structure as input features. Considering that it has been proven that the information about the speaker is more distinct in the lower layers. Because the features extracted by pre-trained wav2vec2 have the ability to distinguish between speakers and this distinction is more obvious in the lower layers, then the weight we will give to the lower layers should be more proportional. So instead of using only representations from the final layer of the pre-trained model, we use a weighted average of representations from all hidden layers to fully exploit the speaker-related information embedded in the entire model.<br/>
      
      In the domain mismatch problem in the language domain, the Farsi data set is Deep Mine data and the source data set is Voxceleb data. For channel domain mismatch, you can use Deep Mine data which includes Persian data in the telephone and wireless field. Also, Farsdat Big data is another data that has good data in the telephone channel.<br/>
      
      For this reason, the domain difference in the speech channel, which can be telephone or wireless, is also considered as two different data sets. that this structure can be considered in the domain difference in other domains of speaker identity confirmation. During the training of a coder, we will have a speaker classification for the source domain and a domain discriminator. The coder tries to extract features from the Persian and English speakers in such a way that the domain discriminator cannot detect them, and in contrast to the domain neural network, it tries to In this mutual or adversary game, win and distinguish between two different domains (for example, the difference in language (English and Farsi) or the difference in the audio channel (telephone or wireless). In fact, the same game is played between the coder and the domain discriminator. In fact We want to have a mapping of the source domain and the target domain that this mapping is almost the same so that the speaker recognition model on the source domain can be matched to the target domain. Finally, when the training process converges, these two maps are similar and the domain discriminator cannot differentiate distinguish between two domains. On the other hand, due to the presence of the speaker classification section, we are sure that the source domain mapping must be able to recognize the speaker, so while the source domain mapping and the target domain mapping are similar, the target domain mapping must also be able to recognize the speaker. has the Then, after training, the coder is responsible for producing the features of the speaker so that these features are invariant to the domain and distinguishing the speaker.<br/>
      
  2. 
      I have divided the project in some levels: <br/>
          Level 1 : Evaluate-Datasets-on-Foundation-Model<br/>
          Level 2 : Finetune-Foundation-Model-with-Datasets<br/>
          Level 3 : SpeakerVerification-simpleway-with-DANN<br/>
          Level 4 : SpeakerVerification-DANN-with-Foundation-Model<br/>
          Level 5 : SpeakerVerification-with-other-new-approaches<br/>
  
  3. 
      BaseLines in articles and sites <br/>
      <br/>

      
 

