import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

them = [0.15669339913633559, 0.15823565700185072, 0.18260333127698952, 0.30876002467612584, 0.58204811844540405, 0.79858112276372606, 0.86027143738433065, 0.92165330043183225, 0.96452806909315236, 0.9438618136952498, 0.92628007402837753, 0.93892658852560151, 0.93183220234423192, 0.92103639728562614, 0.96699568167797656, 0.97624922887106724, 0.91085749537322636, 0.93491671807526222, 0.97717458359037634]
us = [0.52220851326341766, 0.56415792720542879, 0.59037631091918574, 0.61906230721776678, 0.62214682294879708, 0.7208513263417643, 0.79487970388648987, 0.85101789019123997, 0.891425046267736, 0.90653917334978407, 0.92905613818630473, 0.93306600863664402, 0.95157310302282538, 0.90222085132634178, 0.95959284392350397, 0.97378161628624305, 0.86829117828500924, 0.97038864898210986, 0.97100555212831585]
noise_levels = [-18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

plt.figure()
plt.plot(noise_levels, them, label='VTCNN2')
plt.plot(noise_levels, us, label='AlexBasline')
plt.xlabel('Evaluation SNR')
plt.ylabel('Classification Accuracy')
plt.title('Classification Accuracy for Different Evaluation SNRs')
plt.ylim([0,1])
plt.grid()
plt.legend(loc='best')
plt.savefig('them_vs_us.png')





