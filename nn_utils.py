from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


class TrainingHistory(Callback):
    
    def __init__(self, x_test, y_test, CLASSES_LIST):
        super(Callback, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.CLASSES_LIST = CLASSES_LIST
        
    def on_train_begin(self, logs={}):
        self.losses = []
        self.epoch_losses = []
        self.epoch_val_losses = []
        self.val_losses = []
        self.predictions = []
        self.epochs = []
        self.f1 = []
        self.i = 0
        self.save_every = 50

    def on_epoch_end(self, epoch, logs={}):
        
        y_predicted = self.model.predict(self.x_test).argmax(1)
        print(y_predicted.shape)
        
        print("Test Accuracy:", accuracy_score(self.y_test, y_predicted))
    
        p, r, f1, s = precision_recall_fscore_support(self.y_test, y_predicted, 
                                                      average='micro',
                                                      labels=[x for x in 
                                                              self.CLASSES_LIST])
        
        print('p r f1 %.1f %.1f %.1f' % (np.average(p, weights=s)*100.0, 
                                         np.average(r, weights=s)*100.0, 
                                         np.average(f1, weights=s)*100.0))
        
        try:
            print(classification_report(self.y_test, y_predicted, labels=[x for x in 
                                                               self.CLASSES_LIST]))
        except:
            print('ZERO')
            


