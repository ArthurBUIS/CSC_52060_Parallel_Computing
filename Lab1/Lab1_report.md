# Lab 1: Discovering the HPC Software Stack

## Part 1: Discovering the environment

Je travaille sur la machine "loriol", qui contient
- 1 CPU (parce qu'une seule ligne "Package L#")
- 8 coeurs (Core L#0 à L#7)
- 62GB de mémoire libre
- la hiérarchie mémoire est la suivante :
    - Chaque coeur dispose d'une mémoire exclusive 256KB + 32KB + 32KB (soit 320KB)
    - Il y a une mémoire partagée de 16 MB
- network card: "enp0s31f6"


## Part 2: Running a job with slurm

Je compte 171 noeuds disponibles :

```
PARTITION   AVAIL  TIMELIMIT  NODES  STATE NODELIST
SallesInfo*    up 3-00:00:00      1    mix albatros
SallesInfo*    up 3-00:00:00    170   idle ablette,acromion,aerides,ain,allemagne,allier,anchois,angleterre,anguille,apophyse,ardennes,astragale,atlas,autriche,autruche,axis,barbeau,barbue,barlia,baudroie,belgique,bengali,bentley,brochet,bugatti,cadillac,calanthe,carmor,carrelet,charente,cher,chrysler,coccyx,corvette,cote,coucou,creuse,cubitus,cuboide,dindon,diuris,dordogne,doubs,encyclia,epervier,epipactis,espagne,essonne,faisan,femur,ferrari,fiat,finistere,finlande,ford,france,frontal,gardon,gelinotte,gennaria,gironde,groenland,gymnote,habenaria,harpie,hibou,hollande,hongrie,humerus,indre,ipsea,irlande,islande,isotria,jabiru,jaguar,jura,kamiche,labre,lada,landes,lieu,linotte,liparis,lituanie,loire,loriol,lotte,lycaste,malaxis,malleole,malte,manche,marne,maserati,mayenne,mazda,metacarpe,monaco,morbihan,moselle,mouette,mulet,murene,nandou,neotinea,nissan,niva,ombrette,oncidium,ophrys,orchis,parietal,perdrix,perone,peugeot,phalange,piranha,pleione,pogonia,pologne,pontiac,porsche,portugal,quetzal,quiscale,radius,raie,renault,requin,rolls,rotule,rouget,rouloul,roumanie,roussette,rover,royce,sacrum,saone,saumon,serapias,silure,simca,sitelle,skoda,sole,somme,sternum,suede,tarse,telipogon,temporal,test-[252-253],thon,tibia,traquet,truite,urabu,vanda,vanilla,vendee,venturi,verdier,volvo,vosges,xiphoide,xylobium,zeuxine
```

Lorsque j'effectue les commandes, j'obtiens la machine albatros, qui est bien une des machines disponibles. En essayant de nouvelles fois, c'est toujours la machine albatros qui est renvoyée

Avec le script test2, j'obtiens cette fois 3 fois la machine ablette et 1 fois la machine acromion
En essayant d'autres valeurs, j'observe que -n détermine le nombre de processus et -N le nombre de machines différentes sur lesquelles on travaille

J'apercois effectivement le job dans le terminal 2 :

```
(.venv) [loriol Documents]$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
                51 SallesInf sleep_te arthur.b  R       0:08      1 albatros
```
