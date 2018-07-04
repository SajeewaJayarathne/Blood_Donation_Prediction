import csv

writefile = open ("/home/rabbie/Projects/Machine Learning/Driven Data/Blood_Donation_Prediction/output/probability/preprocessed/submission-log-reg-preprocessed-moderated", "w")
with open("/home/rabbie/Projects/Machine Learning/Driven Data/Blood_Donation_Prediction/output/probability/preprocessed/submission-log-reg-preprocessed.csv", "r") as file:
    for line in file:
        vals = line.split(",")
        id = vals[0]
        
        try:
            prob=float(vals[1])
            if (prob <= 0.80):
                if (prob >= 0.20):
                    prob = prob -  0.10
                    # print "nn"
                elif prob >= 0.05 and prob < 0.20:
                    prob = prob -  0.05 
                    # print "dd"
            data = "{},{}\n".format(id, prob)
            #print data
            writefile.write(data)
        except ValueError as shit:
            pass
