import matplotlib.pyplot as plt

# tacc and tloss are lists that contain the training accuracy and loss at each iteration

plt.plot(range(len(tacc)), tacc,'r', label='Training accuracy')

plt.plot(range(len(tloss)), tloss,'b', label='Training loss')

plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Accuracy/Loss')

plt.savefig('train.png',dpi=360)
plt.show()

plt.plot(range(len(vacc)), vacc,'r', label='Valid accuracy')

plt.plot(range(len(vloss)), vloss,'b', label='Valid loss')

plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Accuracy/Loss')

plt.savefig('train.png',dpi=360)
plt.show()
