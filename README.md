Nel seguente File.zip si trova il progetto svolto dal gruppo n° 33 composto da Di Paola Giuseppe, Boudarraja Karim e Nuttini Elena. Il file contiene i seguenti elementi:
CMakeLists.txt:

File di configurazione per CMake che definisce come costruire il progetto. Include la definizione delle variabili di preprocessore per i percorsi delle immagini e dei file di soluzione, e comandi per copiare questi file nella directory di build.
challenge_one.cpp:

File sorgente principale del progetto. Contiene il codice C++ che esegue operazioni di convoluzione su immagini utilizzando matrici sparse di Eigen, e salva i risultati in file immagine e file di testo.
include/:

Directory che contiene i file di intestazione stb_image.h e stb_image_write.h necessari per la gestione delle immagini.
512px-Albert_Einstein_Head.jpeg:

Immagine di esempio utilizzata nel progetto per le operazioni di convoluzione.
soluzioneChallenge.mtx:

File di testo che contiene una matrice di soluzione in formato Matrix Market. Questo file viene caricato e utilizzato nel progetto.
Per l'eseguzione con la libreria di Lis è stato digitato il seguente comando: 

mpirun -n 2 ./test1 convolution_matrix_A2.mtx vector_w.txt soluzioneChallenge.mtx histChallenge.txt -tol 1.0e-9 -maxit 1000 -i bicgstab
