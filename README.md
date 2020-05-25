# webots-kuka-gym-env-master

All'interno della cartella webots_controller è presente sia il controllore da utilizzare su Webots (controller_learning.py), sia tutti i file con i dati dell'apprendimento fatto finora (da inserire, insieme al target_trajectory.npy, nella stessa directory del controllore).

Per continuare l'apprendimento, utilizzando i dati già raccolti, è necessario modficare la varibile continue_learning nel file learning_parameters impostandola a True

Rispetto alla versione precedente è stato aggiornato anche il mondo con l'aggiunta di touch sensors sulle finger del robot. Il file del mondo è presente nella cartella webots_world.

Per l'installazione dell'ambiente gym e del mondo si può fare riferimento alla repository: https://github.com/gabrielesartor/webots-kuka-gym-env
