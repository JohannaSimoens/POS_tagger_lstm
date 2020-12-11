from optparse import OptionParser

def read_corpus(filename, taille_train, taille_test, taille_dev):
    """

    :param filename: corpus à lire 
    :param taille_train: taille du corpus d'apprentissage 
    :param taille_test: taille du corpus de test 
    :param taille_dev: taille du corpus de developpment 
    :return: corpus découpé 
    """
    dictionnaire_phrase = {}
    phrase = ""
    with open(filename, 'r', encoding="utf-8") as fichier:
        for elt in fichier:
            if elt[:16] == "# sentence-text:":
                phrase = elt
                dictionnaire_phrase[phrase] = []
            else:
                new_sequence = elt.split("\t")
                dictionnaire_phrase[phrase].append(new_sequence)
    taille = len(dictionnaire_phrase)
    compteur = 0
    train, test, dev = {}, {}, {}
    for elt in dictionnaire_phrase:
        if compteur < (taille * taille_train):
            train[elt] = dictionnaire_phrase[elt]
        elif compteur < (taille * (taille_train + taille_test)) and compteur >= (taille * taille_train):
            test[elt] = dictionnaire_phrase[elt]
        else:
            dev[elt] = dictionnaire_phrase[elt]
        compteur += 1
    liste_X_train = []
    liste_Y_train = []
    for elt in train:
        liste_x = []
        liste_y = []
        for elt2 in train[elt]:
            if len(elt2) > 4:
                liste_x.append(elt2[1])
                liste_y.append(elt2[3])
        liste_X_train.append(liste_x)
        liste_Y_train.append(liste_y)
    liste_X_test = []
    liste_Y_test = []
    for elt in test:
        liste_x = []
        liste_y = []
        for elt2 in test[elt]:
            if len(elt2) > 4:
                liste_x.append(elt2[1])
                liste_y.append(elt2[3])
        liste_X_test.append(liste_x)
        liste_Y_test.append(liste_y)
    liste_X_dev = []
    liste_Y_dev = []
    for elt in dev:
        liste_x = []
        liste_y = []
        for elt2 in dev[elt]:
            if len(elt2) > 4:
                liste_x.append(elt2[1])
                liste_y.append(elt2[3])
        liste_X_dev.append(liste_x)
        liste_Y_dev.append(liste_y)

    return liste_X_train, liste_Y_train, liste_X_test, liste_Y_test, liste_X_dev, liste_Y_dev

def main():
    """
    main function
    :return: none
    """
    usage = """ %prog [options]"""
    parser = OptionParser(usage=usage, version="%prog version 1.0")

    # Gestion des options et arguments
    parser.add_option("-l",
                      "--taille_train",
                      dest="taille",
                      default=0.8,
                      type=float,
                      help="Taille du train. [default: %default]")
    parser.add_option('-d',
                      '--dataset',
                      dest="dataSet",
                      help='Path to the directory containing the data files. [default: %default]',
                      default="sequoia-7.0/sequoia.deep.conll")
    (options, args) = parser.parse_args()
    taille_train = options.taille
    taille_dev = (1-float(taille_train))/2

    liste_X_train, liste_Y_train, liste_X_test, liste_Y_test, liste_X_dev, liste_Y_dev = read_corpus(
        options.dataSet, taille_train, taille_dev, taille_dev)


if __name__ == '__main__':
    main()
