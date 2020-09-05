# webots-kuka-gym-env-master

Rispetto alla versione precedente è stato aggiornato anche il mondo con l'aggiunta di touch sensors sulle finger del robot. Il file del mondo è presente nella cartella webots_world.

Per l'installazione dell'ambiente gym e del mondo si può fare riferimento alla repository: https://github.com/gabrielesartor/webots-kuka-gym-env

-Funzioni di Reward-
La funzione di reward da utilizzare cambia in base al tipo di azione che deve essere effettuata ed imparata. Per la presa (afferrare l'oggetto) va utilizzata la funzione che si trova alla riga 209 del file webots_kuka_env.py del ramo master, mentre per la posa dell'oggetto va utilizzata la funzione alla riga 208.

-Utilizzare i dati raccolti per le azioni-
All'interno della cartella webots_controller è presente sia il controllore da utilizzare su Webots (/controllers/controller_youbot/controller_youbot.py), sia tutti i file con i dati dell'apprendimento fatto finora (da inserire, insieme al target_trajectory.npy, nella stessa directory del controllore).
I file contenenti i dati raccolti per le prese effettuate si trovano all'interno della cartella /prese, suddivise prima per gli oggetti su cui sono stati effettuati i test di presa, poi per le prese effettuate. (Presa1 -> Afferra l'oggetto; Trasporto1 -> Posa l'oggetto afferrato nel vano).

Per continuare l'apprendimento, utilizzando i dati già raccolti, è necessario modficare la varibile continue_learning nel file learning_parameters.py impostandola a True

-Cambiare i flag-


