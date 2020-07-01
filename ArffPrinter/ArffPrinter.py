class ArffPrinter:

    def __init__(self, n):
        self.n = n

    def initFiles(self):
        effect_filename = self.build_ArffFileName("effects")
        precond_filename = self.build_ArffFileName("preconditions")
        mask_filename = self.build_MaskFileName()
        open(mask_filename, "w") #Apertura che serve a resettare cio' che e' scritto sul file

        with open(effect_filename, "w") as fpE:
            with open(precond_filename, "w") as fpP:

                fpE.write("""@relation effects
@attribute s0 real
@attribute s1 real
@attribute s2 real
@attribute s3 real
@attribute s4 real
@attribute s5 real
@attribute s6 real
@attribute s7 real
@attribute 'class' {false, true}
@data
""")

                fpP.write("""@relation preconditions
@attribute s0 real
@attribute s1 real
@attribute s2 real
@attribute s3 real
@attribute s4 real
@attribute s5 real
@attribute s6 real
@attribute s7 real
@attribute 'class' {false, true}
@data
""")

    def build_ArffFileName(self, suffix):
        return "op" + str(self.n) + "_" + suffix + ".arff"

    def build_MaskFileName(self):
        return "op" + str(self.n) + "_mask.txt"


    ##################################################  METODO DA CHIAMARE PER STAMPARE UNA RIGA SU UNO DEI FILE ARFF #######################################
    def writeArffLine(self, vect, suffix):
        nome_file = self.build_ArffFileName(suffix)
        with open(nome_file, "a") as fp: 
            self.writeValuesEntry(fp, vect)
            if vect[len(vect)-1]:
                fp.write(",true\n")
            else:
                fp.write(",false\n")


    ##################################################  METODO DA CHIAMARE PER STAMPARE UNA RIGA SUL FILE MASK #######################################
    def writeMaskLine(self, vect_prec, vect_eff):
        nome_file = self.build_MaskFileName()
        with open(nome_file, "a") as fp:
            self.writeValuesEntry(fp, vect_prec)
            fp.write(" 0 ")
            self.writeValuesEntry(fp, vect_eff)
            fp.write("\n")


    def writeValuesEntry(self, fp, vect):
        for i in range(0,len(vect)-2):
            n = self.getNumberFrom(vect[i])
            fp.write("%d," %n)
        n = self.getNumberFrom(vect[len(vect)-2])
        fp.write("%d" %n)  #Ultimo valore delle vriabili dei sensori

    def getNumberFrom(self, booleanFlag):
        if booleanFlag:
            return 1
        else:
            return 0
              
#Metodo main usato per una prova della stampa
if __name__ == "__main__":
    printer = ArffPrinter(0)
    printer.initFiles()
    for i in range(0,10):
        array = [True, False, True, True, True, False]
        array1 = [False, False, False, True, True, True]
        printer.writeArffLine(array,"preconditions")
        printer.writeArffLine(array1, "effects")
        printer.writeMaskLine(array, array1)
