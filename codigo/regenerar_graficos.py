"""Regenera figuras a partir de los logs; sin ejecutar de nuevo 25 corridas."""
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Color fijo por configuración en todas las figuras; C4 punteada porque repite a C1 hasta la generación 320.
COLOR={c:plt.get_cmap('tab10').colors[c-1] for c in range(1,6)}
ESTILO={c:('--' if c==4 else '-') for c in range(1,6)}

def build(R):
    with (R/'resultados/log_corridas.csv').open(encoding='utf-8-sig') as f:logs=list(csv.DictReader(f))
    D=defaultdict(list)
    for a in logs:D[(int(a['configuracion']),int(a['replica']))].append(a)
    for cfg in range(1,6):
        fig,ax=plt.subplots(figsize=(9,4.6))
        for rep in range(1,6):
            hist=D[cfg,rep]
            ax.step([int(x['generacion']) for x in hist],[int(x['P_mejor']) for x in hist],where='post',label=f'Ejecución {rep}')
        ax.set(title=f'Configuración {cfg}: evolución de la penalización P',xlabel='Generación',ylabel='P — déficit de cobertura al cuadrado (menor es mejor)')
        ax.grid(alpha=.23);ax.legend(ncol=5,fontsize=8);ax.set_xlim(left=0);fig.tight_layout()
        fig.savefig(R/'graficos'/f'corridas_config_{cfg}.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.1))
    for cfg in range(1,6):
        hist=D[cfg,1]
        ax.step([int(x['generacion']) for x in hist],[int(x['P_mejor']) for x in hist],where='post',label=f'Configuración {cfg}',color=COLOR[cfg],linestyle=ESTILO[cfg])
    ax.set(title='Mejor penalización P a lo largo de las generaciones, misma semilla',xlabel='Generación real',ylabel='Penalización P de cobertura (menor es mejor)')
    ax.grid(alpha=.25);ax.legend(ncol=3);ax.set_xlim(left=0);fig.tight_layout();fig.savefig(R/'graficos'/'comparacion_configuraciones.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.6))
    for cfg in range(1,6):
        hist=D[cfg,1]
        ax.plot([int(x['generacion']) for x in hist],[int(x['P_promedio_generacion']) for x in hist],label=f'C{cfg}',color=COLOR[cfg],linestyle=ESTILO[cfg])
    ax.set(title='Penalización promedio de la población a lo largo de la evolución',xlabel='Generación',ylabel='P medio de la generación')
    ax.grid(alpha=.25);ax.legend(ncol=5);ax.set_xlim(left=0);fig.tight_layout();fig.savefig(R/'graficos'/'penalizacion_poblacion.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.6))
    for cfg in range(1,6):
        hist=D[cfg,1]
        ax.plot([int(x['generacion']) for x in hist],[int(x['individuos_distintos']) for x in hist],label=f'C{cfg}',color=COLOR[cfg],linestyle=ESTILO[cfg])
    ax.set(title='Diversidad genética: cromosomas distintos (réplica 1)',xlabel='Generación',ylabel='Cantidad de individuos únicos')
    ax.grid(alpha=.25);ax.legend(ncol=5);ax.set_xlim(left=0);fig.tight_layout();fig.savefig(R/'graficos'/'diversidad_poblacion.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.6))
    for cfg in range(1,6):
        hist=D[cfg,1]
        feasible=[r for r in hist if int(r['P_mejor'])==0]
        if feasible:
            ax.step([int(x['generacion']) for x in feasible],
                    [int(x['preferencias_mejor']) for x in feasible],
                    where='post',label=f'C{cfg}',color=COLOR[cfg],linestyle=ESTILO[cfg])
    ax.set(title='Evolución del costo Q desde la primera solución con cobertura completa',
           xlabel='Generación real',ylabel='Preferencias Q (menor es mejor)')
    ax.grid(alpha=.25);ax.legend();ax.set_xlim(left=0);fig.tight_layout()
    fig.savefig(R/'graficos'/'preferencias_desde_factibilidad.png',dpi=180);plt.close(fig)
    # Holgura (disponibles - requeridos) del mejor cronograma por grupo y semana.
    with (R/'resultados/cobertura_mejor_cronograma.csv').open(encoding='utf-8-sig') as f:cov=list(csv.DictReader(f))
    grupos=list(dict.fromkeys(r['grupo'] for r in cov))
    H=[[0]*52 for _ in grupos]
    for r in cov:H[grupos.index(r['grupo'])][int(r['semana'])-1]=int(r['disponibles'])-int(r['requeridos'])
    fig,ax=plt.subplots(figsize=(11,4.2))
    im=ax.imshow(H,aspect='auto',cmap='Blues',vmin=0,extent=(0.5,52.5,len(grupos)-0.5,-0.5))
    ax.set_yticks(range(len(grupos)));ax.set_yticklabels(grupos)
    ax.set(title='Mejor cronograma: holgura semanal (disponibles − requeridos); 0 = cobertura justa, nunca negativa',xlabel='Semana')
    fig.colorbar(im,ax=ax,label='Personas de holgura');fig.tight_layout()
    fig.savefig(R/'graficos'/'holgura_mejor_cronograma.png',dpi=180);plt.close(fig)
    print('Gráficos corregidos en',R/'graficos')

if __name__=="__main__":
    build(Path(__file__).resolve().parent.parent)
