{\rtf1\ansi\ansicpg1252\cocoartf2869
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 flowchart LR\
    subgraph USER["User-defined"]\
        PCFG["population_cfg"]\
        SCFG["species defaults / overrides"]\
        VCFG["var_cfg"]\
    end\
\
    subgraph BUILD["Builders"]\
        PB["PopulationBuilder"]\
        VB["VariableBuilder"]\
    end\
\
    subgraph CORE["Core structures"]\
        POP["ParticlePopulation"]\
        PART["Particle"]\
    end\
\
    subgraph ANALYSIS["Analysis structures"]\
        PVAR["PopulationVariable"]\
        TVAR["ParticleVariable"]\
    end\
\
    subgraph DIAG["Diagnostics"]\
        PDIAG["population-level diagnostics"]\
        TDIAG["particle-level diagnostics"]\
    end\
\
    PCFG --> PB\
    SCFG --> PB\
    PB --> POP\
\
    POP -->|"get_particle"| PART\
    PART -->|"set_particle"| POP\
\
    VCFG --> VB\
    POP --> PVAR\
    PART --> TVAR\
    VB --> PVAR\
    VB --> TVAR\
\
    PVAR --> PDIAG\
    TVAR --> TDIAG}