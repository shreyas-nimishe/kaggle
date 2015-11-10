from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from numpy import genfromtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    #f8 means float
    dataset = genfromtxt(open('./cs-training.csv','r'), delimiter=',', dtype='f8')[1:] 
    #target = [x[0] for x in dataset]
    #train = [x[1:] for x in dataset]
    target = dataset[:,1]
    train = dataset[:,2:]		
    test = genfromtxt(open('./cs-test.csv','r'), delimiter=',', dtype='f8')[1:,2:]
    print "dataset made using numpy function genfromtxt"

    #imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
    #new_train = imp.fit_transform(train)
    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    print len(train)
    rf = RandomForestClassifier(n_estimators=15,n_jobs=4)
    rf.fit(train, target)
    print "data fitted using meathod fit in RandomForestClassifier"
    
   
    predicted_probs = [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(test))]
    print "probability calculated"

    savetxt('./submission.csv', predicted_probs, delimiter=',', fmt='%d,%f', 
            header='Id,Probability', comments = '')
   
if __name__ == "__main__":
    main()
