"""Validador independiente de las corridas y cronogramas CSV."""
import csv
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent

def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def verify():
    employees=rows(ROOT/'datos'/'empleados_caso.csv')
    demands=rows(ROOT/'datos'/'demanda_semanal.csv')
    logs=rows(ROOT/'resultados'/'log_corridas.csv')
    scores=rows(ROOT/'resultados'/'resumen_corridas.csv')
    schedule=rows(ROOT/'resultados'/'cronogramas_25_corridas.csv')
    greedy=rows(ROOT/'resultados'/'comparacion_voraz.csv')
    assert len(employees)==500 and len(demands)==9*52
    assert len(scores)==25 and len(greedy)==5
    assert len(schedule)==25*500
    E={e['empleado_id']:e for e in employees}
    D={(d['grupo'],int(d['semana'])):int(d['personas_requeridas']) for d in demands}
    assert all(1<=w<=52 for _,w in D)
    S={(int(x['configuracion']),int(x['replica'])):x for x in scores}
    byrun=defaultdict(list)
    for x in schedule:byrun[(int(x['configuracion']),int(x['replica']))].append(x)
    assert len(byrun)==25
    out=[]
    for key,series in byrun.items():
        assert len(series)==500 and {r['empleado_id'] for r in series}==set(E)
        counts=defaultdict(int);prefs=0
        for r in series:
            e=E[r['empleado_id']]
            vac=sorted(int(w) for w in r['semanas_vacaciones'].split(';'))
            assert len(vac)*7==int(e['dias_vacaciones'])
            assert len(set(vac))==len(vac)
            assert min(vac)>=1 and max(vac)<=52
            tech=int(e['equipo_tecnico'])==1
            if tech: assert 1 not in vac
            else: assert 1 in vac
            a=int(r['inicio_bloque_variable']);n=int(e['dias_vacaciones'])//7
            should=list(range(a,a+(n if tech else n-1)))
            if not tech:should=[1]+should
            assert sorted(should)==vac
            prefs+=abs(a-int(e['semana_preferida']))*int(e['prioridad_preferencia'])
            for w in vac:counts[(e['grupo'],w)]+=1
        group_sizes=defaultdict(int)
        for e in employees:group_sizes[e['grupo']]+=1
        gaps=[max(0, D[(g,w)]-(group_sizes[g]-counts[g,w])) for g,w in D]
        p=sum(v*v for v in gaps);u=sum(gaps);m=max(gaps)
        expected=S[key]
        assert (p,u,m,prefs)==(int(expected['P_final']),int(expected['U_final']),int(expected['M_final']),int(expected['preferencias'])),(key,p,u,m,prefs)
        assert int(expected['factible'])==int(p==0)
        out.append((key,p,u,m,prefs))
    bylog=defaultdict(list)
    for x in logs:bylog[(int(x['configuracion']),int(x['replica']))].append(x)
    assert set(bylog)==set(byrun)
    for key,history in bylog.items():
        generations=int(S[key]['generaciones'])
        assert len(history)==generations+1
        assert [int(v['generacion']) for v in history]==list(range(generations+1))
        # best-so-far fitness must be nonincreasing lexicographically
        fit=[tuple(int(x[k]) for k in ('P_mejor','U_mejor','M_mejor','preferencias_mejor')) for x in history]
        assert all(fit[i]<=fit[i-1] for i in range(1,len(fit)))
        assert fit[-1]==(int(S[key]['P_final']),int(S[key]['U_final']),int(S[key]['M_final']),int(S[key]['preferencias']))
    # Baseline rerun independent reconstruction from exported weeks.
    for b in greedy:
        rep=int(b['replica'])
        sr=rows(ROOT/'resultados'/f'voraz_{rep}.csv')
        assert len(sr)==500
        counts=defaultdict(int);prefs=0
        for r in sr:
            e=E[r['empleado_id']];start=int(r['semana_inicio_variable'])
            tech=int(e['equipo_tecnico'])==1
            duration=int(e['dias_vacaciones'])//7
            for w in list(range(start,start+(duration if tech else duration-1)))+([] if tech else [1]):counts[(e['grupo'],w)]+=1
            prefs+=int(e['prioridad_preferencia'])*abs(start-int(e['semana_preferida']))
        group_sizes=defaultdict(int)
        for e in employees:group_sizes[e['grupo']]+=1
        gaps=[max(0,D[(g,w)]-(group_sizes[g]-counts[g,w])) for g,w in D]
        assert (sum(v*v for v in gaps),sum(gaps),max(gaps),prefs)==tuple(int(b[k]) for k in ('P','U','M','preferencias'))
    print(f'VALIDACIÓN CORRECTA: {len(scores)} corridas, {len(schedule)} asignaciones, {len(demands)} restricciones perfil-semana, {len(logs)} líneas de evolución y {len(greedy)} corridas voraces.')
    print('Corridas con cobertura completa:',sum(p==0 for _,p,_,_,_ in out),'/25')

if __name__=='__main__': verify()
