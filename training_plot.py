import matplotlib.pyplot as plt

fname = 'training_history.txt'
text_file = open(fname, "r")
lines = text_file.readlines()

loss = []
val_loss = []

for line in lines:
	# 39792/39792 [==============================] - 88s - loss: 0.0243 - val_loss: 0.0369
	if '=====' in line:
		tmp = line.split(':')
		loss.append(float((tmp[1].split('-')[0]).strip()))
		val_loss.append(float(tmp[2].strip()))
		# print('loss: {}, val loss: {}'.format(loss[-1], val_loss[-1]))
		
Epoch = range(1, len(loss)+1)
	
fig = plt.figure(figsize=(8,4))

ax = fig.add_subplot(1, 1, 1)
ax.plot(Epoch, loss, 'g', Epoch, val_loss, 'b')
ax.set_title('Behavior Cloning CNN')
ax.legend(('training', 'validation'))
ax.set_xlim([1, 50])
ax.set_ylim([0, 0.04])
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')


fig.savefig('training_history')

# display the plot
plt.show()