#!/usr/bin/env python3
"""TP2 UTN-FRBA G15. AG para vacaciones, escenario experimental reproducible.
No es un sistema de planificación laboral real ni incorpora normativa laboral completa.
"""
from __future__ import annotations
import argparse, csv, math, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

N, W = 500, 52
DATA_SEED = 20261020
GROUPS = ('Mecanica','Electricidad','Soldadura','Hidraulica','Instrumentacion',
          'Produccion','Envasado','Logistica','Calidad')
GROUP_SIZES = (20,20,20,20,20,140,110,90,60)
SEEDS = (1101,2202,3303,4404,5505)
# Cinco configuraciones, cinco replicas cada una. La prueba de sensibilidad cambia un factor principal.
CONFIGS = (
 dict(id=1,poblacion=64,generaciones=320,mutacion=0.85,elitismo=3,torneo=3,cruce=0.90),
 dict(id=2,poblacion=64,generaciones=320,mutacion=0.45,elitismo=3,torneo=3,cruce=0.90),
 dict(id=3,poblacion=96,generaciones=320,mutacion=0.85,elitismo=3,torneo=3,cruce=0.90),
 dict(id=4,poblacion=64,generaciones=450,mutacion=0.85,elitismo=3,torneo=3,cruce=0.90),
 dict(id=5,poblacion=64,generaciones=320,mutacion=0.85,elitismo=8,torneo=3,cruce=0.90),
)
@dataclass(frozen=True)
class Empleado:
    id: str
    grupo: int
    tecnico: bool
    antiguedad: int
    dias: int
    preferida: int
    prioridad: int

def dias(anios):
    # Tramos simplificados del caso; no reemplazan asesoramiento laboral.
    return 14 if anios<5 else 21 if anios<10 else 28 if anios<20 else 35

def crear_datos():
    rng=np.random.default_rng(DATA_SEED)
    empleados=[]
    grupos=np.repeat(np.arange(len(GROUPS)),GROUP_SIZES)
    assert len(grupos)==N
    for idx,g in enumerate(grupos):
        tech=int(g)<5
        antiguedad=int(rng.choice([2,7,14,24],p=[.39,.33,.22,.06]))
        dur=dias(antiguedad)
        # Preferencias con concentracion en tres ventanas de alta competencia
        ventana=int(rng.choice((10,25,45),p=(.30,.45,.25)))
        preferida=int(np.clip(ventana+rng.integers(-3,4),2,52-dur//7+1))
        prioridad=int(rng.choice((1,2,3),p=(.50,.35,.15)))
        empleados.append(Empleado(f'E{idx+1:03d}',int(g),tech,antiguedad,dur,preferida,prioridad))
    # Semanas de máxima exigencia: mantenimiento de semana 1 y picos productivos variados.
    pedidos=np.zeros((len(GROUPS),W),dtype=np.int16)
    for g,n in enumerate(GROUP_SIZES):
        for w in range(1,W+1):
            if g>=5 and w==1: pedidos[g,w-1]=0;continue # parada de produccion
            if g<5 and w==1: pedidos[g,w-1]=n;continue # mantenimiento
            estacional=.035*math.sin(2*math.pi*(w-1)/W+g*.53)
            # Certificados: 1-3 ausencias posibles por semana; operativos: ~5%-9%
            relax=(0.067+estacional) if g<5 else (.075+estacional)
            cupo=max(1,round(n*relax))
            if (w+3*g)%17 in (0,1) or w in (11,26,46):
                cupo=max(0,cupo-1 if g<5 else cupo-2)
            pedidos[g,w-1]=n-cupo
    return empleados,pedidos

class Problema:
    def __init__(self,emps,demanda):
        self.emps=emps;self.demanda=demanda
        self.n=len(emps);self.G=len(GROUPS)
        self.grupo=np.asarray([x.grupo for x in emps],np.int16)
        self.tecnico=np.asarray([x.tecnico for x in emps],bool)
        self.dur=np.asarray([x.dias//7 for x in emps],np.int16)
        self.len=np.where(self.tecnico,self.dur,self.dur-1).astype(np.int16)
        self.preferida=np.asarray([x.preferida for x in emps],np.int16)
        self.prioridad=np.asarray([x.prioridad for x in emps],np.int16)
        self.valid_starts=[np.arange(2,W-int(self.len[i])+2,dtype=np.int16) for i in range(self.n)]
        self.maxstarts=np.asarray([x[-1] for x in self.valid_starts],np.int16)
        self.primer_semana=np.bincount(self.grupo[~self.tecnico],minlength=self.G)
    def random(self,rng, cerca=True):
        uniform=rng.integers(2,self.maxstarts+1,size=self.n,dtype=np.int16)
        if not cerca:return uniform
        wish=np.clip(self.preferida+rng.integers(-8,9,size=self.n),2,self.maxstarts)
        return np.where(rng.random(self.n)<.66,wish,uniform).astype(np.int16)
    def absences(self,starts):
        delta=np.zeros((self.G,W+1),dtype=np.int16)
        np.add.at(delta,(self.grupo,starts-1),1)
        np.add.at(delta,(self.grupo,starts-1+self.len),-1)
        out=np.cumsum(delta,axis=1)[:,:W]
        out[:,0]+=self.primer_semana
        return out
    def score(self,starts):
        fuera=self.absences(starts)
        disponible=np.asarray(GROUP_SIZES)[:,None]-fuera
        gap=np.maximum(self.demanda-disponible,0)
        penalty=int(np.square(gap.astype(np.int32)).sum())
        missing=int(gap.sum())
        maxgap=int(gap.max())
        # costo de preferencias (soft): magnitud de desplazamiento ponderado
        delay=np.abs(starts.astype(np.int32)-self.preferida)
        preference=int(np.dot(delay,self.prioridad.astype(np.int32)))
        return (penalty,missing,maxgap,preference)
    def decode(self,starts):
        w=np.arange(1,W+1)[None,:]
        off=(w>=starts[:,None])&(w<starts[:,None]+self.len[:,None])
        off[~self.tecnico,0]=True
        return off
    def check(self,starts):
        if not len(starts)==N:raise AssertionError('n')
        if not np.all((starts>=2)&(starts<=self.maxstarts)):raise AssertionError('inicios fuera de rango')
        off=self.decode(starts)
        assert np.array_equal(off.sum(axis=1),self.dur)
        assert not off[self.tecnico,0].any()
        assert off[~self.tecnico,0].all()
        assert np.array_equal(self.absences(starts),np.vstack([off[self.grupo==g].sum(axis=0) for g in range(self.G)]))
        s=self.score(starts); assert s[0]>=0 and s[3]>=0
        return s
    def demand_rows(self,starts):
        absences=self.absences(starts);disp=np.asarray(GROUP_SIZES)[:,None]-absences
        return [(GROUPS[g],w+1,int(self.demanda[g,w]),int(disp[g,w]),int(max(0,self.demanda[g,w]-disp[g,w]))) for g in range(self.G) for w in range(W)]

def select_tournament(fitness,rng,k):
    pick=rng.integers(len(fitness),size=k)
    return min((int(i) for i in pick),key=lambda ix:fitness[ix])

def ga(p,cfg,seed):
    """AG autónomo: sus individuos iniciales NO provienen del método voraz.

    Representación: un gen de inicio por empleado, decodificable a 500x52.
    Selección por torneo, cruce por grupos enteros, mutación dirigida al déficit
    o a preferencias, intercambio de semanas y elitismo.
    """
    rng=np.random.default_rng(seed)
    population=[p.random(rng) for _ in range(cfg['poblacion'])]
    subsets=[np.flatnonzero(p.grupo==g) for g in range(p.G)]
    start=time.perf_counter();best=None;history=[]
    for gen in range(cfg['generaciones']+1):
        fitness=[p.score(x) for x in population]
        order=sorted(range(len(population)),key=lambda i:fitness[i])
        if best is None or fitness[order[0]]<best[0]:
            best=(fitness[order[0]],population[order[0]].copy())
        history.append((gen,*best[0],int(np.mean([f[0] for f in fitness])),
                        len({x.tobytes() for x in population})))
        if gen==cfg['generaciones']:break
        nextpop=[population[i].copy() for i in order[:cfg['elitismo']]]
        while len(nextpop)<len(population):
            a=population[select_tournament(fitness,rng,cfg['torneo'])]
            b=population[select_tournament(fitness,rng,cfg['torneo'])]
            child=a.copy()
            if rng.random()<cfg['cruce']:
                # Cruce de grupos completos: combina perfiles evolucionados
                # sin perder la estructura de vacaciones por empleado.
                for indices in subsets:
                    if rng.random()<.5:child[indices]=b[indices]
            if rng.random()<cfg['mutacion']:
                for _ in range(int(rng.integers(1,5))):
                    # Mutación orientada al problema, SIN resolverlo por voraz:
                    # identifica faltantes para intentar desplazar una ausencia.
                    absn=p.absences(child)
                    gap=np.maximum(p.demanda-(np.asarray(GROUP_SIZES)[:,None]-absn),0)
                    if gap.any() and rng.random()<.90:
                        g,w=map(int,divmod(int(np.argmax(gap)),W))
                        ids=subsets[g]
                        busy=ids[((w+1>=child[ids]) &
                                  (w+1<child[ids]+p.len[ids])) |
                                 ((w==0)&(~p.tecnico[ids]))]
                        if len(busy)==0:continue
                        i=int(rng.choice(busy))
                        opts=p.valid_starts[i][~((w+1>=p.valid_starts[i]) &
                                                  (w+1<p.valid_starts[i]+p.len[i]))]
                        if len(opts):
                            # Dos variantes de mutación sobre inicios permitidos.
                            child[i]=int(opts[np.argmin(np.abs(opts-p.preferida[i]))]) if rng.random()<.5 else int(rng.choice(opts))
                    else:
                        # Si hay cobertura, se evoluciona también la calidad Q.
                        weight=np.abs(child.astype(np.int32)-p.preferida)*p.prioridad
                        i=int(rng.choice(p.n,p=weight/weight.sum())) if weight.sum() and rng.random()<.75 else int(rng.integers(p.n))
                        target=int(np.clip(p.preferida[i]+rng.integers(-3,4),2,p.maxstarts[i]))
                        child[i]=target if rng.random()<.8 else int(rng.choice(p.valid_starts[i]))
            if rng.random()<.35:
                # Mutación de intercambio no guiada por voraz ni por prueba
                # de candidatos: intercambia genes de igual grupo/duración.
                i=int(rng.integers(p.n));same=subsets[int(p.grupo[i])]
                partners=same[p.len[same]==p.len[i]]
                if len(partners)>1:
                    j=int(rng.choice(partners));child[i],child[j]=child[j],child[i]
            nextpop.append(child)
        population=nextpop
    return best,history,time.perf_counter()-start

def greedy(p,seed):
    # Voraz comparado bajo la MISMA función objetivo: cobertura dura primero y preferencias después.
    # Sin revisión de empleados ya asignados (precisamente la limitación constructiva estudiada).
    rng=np.random.default_rng(seed)
    st=time.perf_counter()
    # Para cada empleado se evalua el costo marginal de insertar vacaciones.
    # Antes de ubicar a un empleado se considera al resto aún sin vacaciones.
    current=np.zeros((p.G,W),dtype=np.int16)
    current[:,0]=p.primer_semana
    starts=np.zeros(N,dtype=np.int16)
    sizes=np.array(GROUP_SIZES)[:,None]
    order=rng.permutation(N)
    for idx in order:
        i=int(idx);g=int(p.grupo[i]);length=int(p.len[i])
        options=p.valid_starts[i].copy();rng.shuffle(options)
        rank=None;selected=None
        for s in options:
            sl=slice(int(s)-1,int(s)-1+length)
            current[g,sl]+=1
            gap=np.maximum(0,p.demanda[g]- (sizes[g,0]-current[g]))
            # compara penalización total del perfil actual mas costo de la preferencia
            score=(int(np.square(gap.astype(np.int32)).sum()),int(gap.sum()),int(gap.max()),
                   int(p.prioridad[i]*abs(int(s)-int(p.preferida[i]))))
            current[g,sl]-=1
            if rank is None or score<rank:rank=score;selected=int(s)
        starts[i]=selected
        current[g,selected-1:selected-1+length]+=1
    return p.check(starts),starts,time.perf_counter()-st

def save_csv(path,header,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)

def render_figures(out,all_logs,summaries,p,champion):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=plt.get_cmap('tab10').colors
    for c in CONFIGS:
        fig,ax=plt.subplots(figsize=(9.7,4.4))
        for rep in range(1,6):
            hist=all_logs[(c['id'],rep)]
            ax.step([r[0] for r in hist],[r[1] for r in hist],where='post',label=f'Repetición {rep}',alpha=.8)
        ax.set(xlabel='Generación',ylabel='Penalización de cobertura $P$ (menor es mejor)',
               title=f'Configuración {c["id"]}: evolución de aptitud en cinco ejecuciones')
        ax.set_xlim(0,c['generaciones']);ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(ncol=5,fontsize=8)
        fig.tight_layout();fig.savefig(out/'graficos'/f'corridas_config_{c["id"]}.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.7))
    for c in CONFIGS:
        hist=all_logs[(c['id'],1)]
        ax.step([r[0] for r in hist],[r[1] for r in hist],where='post',label=f'Config. {c["id"]}',alpha=.87)
    ax.set(xlabel='Generación (eje real)',ylabel='Déficit cuadrático P',title='Comparación entre configuraciones (misma semilla, réplica 1)')
    ax.grid(alpha=.2);ax.set_ylim(bottom=0);ax.set_xlim(0,max(c['generaciones'] for c in CONFIGS));ax.legend(ncol=5)
    fig.tight_layout();fig.savefig(out/'graficos'/'comparacion_configuraciones.png',dpi=160);plt.close(fig)
    # Dos objetivos en paneles independientes para evitar mezcla de unidades.
    fig,ax=plt.subplots(figsize=(10,4.5))
    for c in CONFIGS:
        ax.scatter([c['id']]*5,[r['preferencias'] for r in summaries if r['configuracion']==c['id']],alpha=.75,label=f'C{c["id"]}')
    ax.set(xlabel='Configuración',ylabel='Costo de preferencias (menor es mejor)',title='Distribución de preferencias finales en 25 ejecuciones')
    ax.set_xticks(range(1,6));ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'graficos'/'preferencias_dispersion.png',dpi=155);plt.close(fig)
    rows=p.demand_rows(champion)
    fig,ax=plt.subplots(figsize=(10,4.4))
    for g in GROUPS:
        part=[r for r in rows if r[0]==g]; ax.plot([r[1] for r in part],[r[4] for r in part],label=g)
    ax.set(xlabel='Semana',ylabel='Personas faltantes',title='Faltantes por grupo: mejor cronograma')
    ax.set_xlim(1,52);ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=7)
    fig.tight_layout();fig.savefig(out/'graficos'/'faltantes_ganador.png',dpi=160);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent.parent)
    a=ap.parse_args();out=a.out.resolve();[ (out/sub).mkdir(parents=True,exist_ok=True) for sub in ('datos','resultados','graficos')]
    emps,demanda=crear_datos(); p=Problema(emps,demanda)
    save_csv(out/'datos'/'empleados_caso.csv',('empleado_id','grupo','equipo_tecnico','antiguedad_anios','dias_vacaciones','semana_preferida','prioridad_preferencia'),
             ((e.id,GROUPS[e.grupo],int(e.tecnico),e.antiguedad,e.dias,e.preferida,e.prioridad) for e in emps))
    save_csv(out/'datos'/'demanda_semanal.csv',('grupo','semana','personas_requeridas'),
             ((GROUPS[g],w+1,int(demanda[g,w])) for g in range(len(GROUPS)) for w in range(W)))
    rows_logs=[];rows_scores=[];rows_params=[];rows_calendars=[];alllogs={};baseline=[]
    best_global=None
    for rep,seed in enumerate(SEEDS,1):
        fitness,starts,dt=greedy(p,seed)
        baseline.append(dict(replica=rep,semilla=seed,P=fitness[0],U=fitness[1],M=fitness[2],preferencias=fitness[3],segundos=round(dt,3)))
        save_csv(out/'resultados'/f'voraz_{rep}.csv',('empleado_id','semana_inicio_variable'),((emps[i].id,int(starts[i])) for i in range(N)))
    for cfg in CONFIGS:
        rows_params.append((cfg['id'],cfg['poblacion'],cfg['generaciones'],cfg['mutacion'],cfg['elitismo'],cfg['torneo'],cfg['cruce'],'aleatoria_admisible_sin_voraz'))
        for rep,seed in enumerate(SEEDS,1):
            winner,history,dt=ga(p,cfg,seed)
            p.check(winner[1]); alllogs[cfg['id'],rep]=history
            for h in history:
                rows_logs.append((cfg['id'],rep,seed,*h))
            score=winner[0];initial=history[0]
            rows_scores.append(dict(configuracion=cfg['id'],replica=rep,semilla=seed,poblacion=cfg['poblacion'],generaciones=cfg['generaciones'],mutacion=cfg['mutacion'],elitismo=cfg['elitismo'],P_inicial=initial[1],Q_inicial=initial[4],P_final=score[0],U_final=score[1],M_final=score[2],preferencias=score[3],factible=int(score[0]==0),segundos=round(dt,3)))
            decoded=p.decode(winner[1])
            for i,e in enumerate(emps):
                rows_calendars.append((cfg['id'],rep,e.id,GROUPS[e.grupo],e.dias,int(winner[1][i]),';'.join(map(str,np.flatnonzero(decoded[i])+1))))
            if best_global is None or score<best_global[0]:best_global=(score,winner[1].copy(),cfg['id'],rep)
            print(f"C{cfg['id']}-R{rep} P inicial={initial[1]} P final={score[0]} U={score[1]} Q={score[3]} t={dt:.1f}s",flush=True)
    s,starts,cfg,rep=best_global;assert s==p.check(starts)
    save_csv(out/'resultados'/'parametros_corridas.csv',('configuracion','poblacion','generaciones','prob_mutacion_por_hijo','elitismo','torneo_k','prob_cruce','inicializacion'),rows_params)
    save_csv(out/'resultados'/'log_corridas.csv',('configuracion','replica','semilla','generacion','P_mejor','U_mejor','M_mejor','preferencias_mejor','P_promedio_generacion','individuos_distintos'),rows_logs)
    cols=('configuracion','replica','semilla','poblacion','generaciones','mutacion','elitismo','P_inicial','Q_inicial','P_final','U_final','M_final','preferencias','factible','segundos')
    save_csv(out/'resultados'/'resumen_corridas.csv',cols,([x[key] for key in cols] for x in rows_scores))
    save_csv(out/'resultados'/'comparacion_voraz.csv',('replica','semilla','P','U','M','preferencias','segundos'),([x[k] for k in ('replica','semilla','P','U','M','preferencias','segundos')] for x in baseline))
    save_csv(out/'resultados'/'cronogramas_25_corridas.csv',('configuracion','replica','empleado_id','grupo','dias','inicio_bloque_variable','semanas_vacaciones'),rows_calendars)
    off=p.decode(starts)
    save_csv(out/'resultados'/'mejor_cronograma.csv',('empleado_id','grupo','dias_vacaciones','semana_preferida','inicio_bloque_variable','semanas_vacaciones'),
             ((e.id,GROUPS[e.grupo],e.dias,e.preferida,int(starts[i]),';'.join(map(str,np.flatnonzero(off[i])+1))) for i,e in enumerate(emps)))
    save_csv(out/'resultados'/'cobertura_mejor_cronograma.csv',('grupo','semana','requeridos','disponibles','faltantes'),p.demand_rows(starts))
    render_figures(out,alllogs,rows_scores,p,starts)
    from regenerar_graficos import build as rebuild_fitness_graphs
    rebuild_fitness_graphs(out)
    counts=[]
    for c in CONFIGS:
        sub=[x for x in rows_scores if x['configuracion']==c['id']]
        counts.append(dict(config=c['id'],factibles=sum(x['factible'] for x in sub),P_mean=np.mean([x['P_final'] for x in sub]),P_std=np.std([x['P_final'] for x in sub],ddof=1),prefs_mean=np.mean([x['preferencias'] for x in sub]),time_mean=np.mean([x['segundos'] for x in sub])))
    print('\nGANADOR',cfg,rep,s,'VORAZ',baseline,flush=True)
    for c in counts:print('RESUMEN',c,flush=True)
if __name__=='__main__':main()
