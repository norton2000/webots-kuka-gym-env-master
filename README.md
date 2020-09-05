# webots-kuka-gym-env-master
-Installazione-
Rispetto alla versione precedente è stato aggiornato anche il mondo con l'aggiunta di touch sensors sulle finger del robot. Il file del mondo è presente nella cartella webots_world.

Per l'installazione dell'ambiente gym e del mondo si può fare riferimento alla repository: https://github.com/gabrielesartor/webots-kuka-gym-env

NOTA: Le righe dei file indicate in questa guida fanno riferimento ai file dopo la conclusione del progetto del tirocinio, prima della consegna.

-Funzioni di Reward-
La funzione di reward da utilizzare cambia in base al tipo di azione che deve essere effettuata ed imparata. Per la presa (afferrare l'oggetto) va utilizzata la funzione che si trova alla riga 209 del file webots_kuka_env.py del ramo master, mentre per la posa dell'oggetto va utilizzata la funzione alla riga 208.

-Decidere l'oggetto di cui effettuare la presa-
Se si vuole imparare la presa di un oggetto presente nel mondo e posto davanti al robot, per prima cosa bisogna assicurarsi che nell'albero degli elementi del mondo di webots l'oggetto compaia come un nodo che ha una definizione "DEF" con un dato nome. (es. "ball", "ring"). Per decidere di quale oggetto si vuole effettuare la presa, bisogna modificare l'array "objects" nel file controller_youbot.py:25, mettendo come primo elemento una stringa con il nome dell'oggetto da prendere (quello inserito nel campo DEF del nodo di webots) e come secondo elemento l'altro oggetto che prende parte alla simulazione.

-Utilizzare i dati raccolti per le azioni-
All'interno della cartella webots_controller è presente sia il controllore da utilizzare su Webots (/controllers/controller_youbot/controller_youbot.py), sia tutti i file con i dati dell'apprendimento fatto finora (da inserire, insieme al target_trajectory.npy, nella stessa directory del controllore).
I file contenenti i dati raccolti per le prese effettuate si trovano all'interno della cartella /Prese, suddivise prima per gli oggetti su cui sono stati effettuati i test di presa, poi per le prese effettuate. (Presa1 -> Afferra l'oggetto; Trasporto1 -> Posa l'oggetto afferrato nel vano).


-Cambiare i flag-
Per effettuare correttamente la sperimentazione dell'apprendimento delle azioni e la raccolta dei dati (file .arff) è necessario utilizzare i seguenti flag nel modo seguente:

--> "continue_learning" (file learning_parameters.py:43): questo flag indica se il robot deve effettuare l'azione apprendendola dall'inizio, o se deve continuare l'apprendimento basandosi sui risultati degli episodi precedenti. Impostarla a False se si vuole apprendere l'azione dall'inizio (il robot non la conosce e deve impararla). Impostarla a True per continuare l'apprendimento dopo che la simulazione ha concluso il ciclo di epoche. Nota: quando si vuole portare il robot da conoscenza nulla di un'azione, fino ad apprenderla correttamente, se sono necessari più cicli di epoche, allora bisogna impostare questo flag a False per il primo ciclo, ed a True dal successivo in poi.

--> "write_arff_files" (file learning_parameters.py:44): questo flag indica se bisogna o meno scrivere i file arff durante la simulazione.
--> "continue_writing" (file option_classifiers.py:16): questo flag indica se, nello scrivere i file arff, bisogna continuare a scrivere in append su dei file preesistenti (True) oppure se creare dei nuovi file cancellando eventuali file preesistenti (False).

-Salvare il mondo tra un'azione ed un'altra-
Se si vuole eseguire l'azione di posa dopo l'azione di presa, bisonga utilizzare, come mondo di partenza per la posa, la situazione del mondo alla fine dell'esecuzione della presa. Per tale motivo, una volta eseguita la presa, prima che il mondo venga resettato bisogna salvarlo ed utilizzarlo per l'azione di posa per quell'oggetto.
