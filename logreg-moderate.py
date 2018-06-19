import csv

writefile = open ("/home/sajeewa/Desktop/Blood Donation ML/Blood_Donation_Prediction/output/probability/new_shit_yoo.csv", "w")
with open("/home/sajeewa/Downloads/submission-log-reg.csv", "r") as file:
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
            print data
            writefile.write(data)
        except ValueError as shit:
            pass
    
        # print prob
        # print id
        # print prob
        
            # print prob
       
        # print data
        
        # file.write(id, prob)
        
