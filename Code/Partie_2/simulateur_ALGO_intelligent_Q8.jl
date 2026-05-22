using SimJulia, Distributions, Random, Statistics, ResumableFunctions, Printf
using Plots

# --- DONNÉES OFFICIELLES ---
const λ = [0.29, 0.32, 0.47, 0.38]
const PARCOURS = [
    [1, 2, 3, 4, 8], # T1
    [2, 4, 7],       # T2
    [3, 5, 1],       # T3
    [5, 6, 7, 8]     # T4
]
const TEMPS = Dict(
    (1,1)=>(0.58,0.78), (1,2)=>(0.23,0.56), (1,3)=>(0.81,0.93), (1,4)=>(0.12,0.39), (1,8)=>(0.82,1.04),
    (2,2)=>(0.59,0.68), (2,4)=>(0.74,0.77), (2,7)=>(0.30,0.55),
    (3,3)=>(0.37,0.54), (3,5)=>(0.35,0.63), (3,1)=>(0.57,0.64),
    (4,5)=>(0.36,0.51), (4,6)=>(0.61,0.70), (4,7)=>(0.78,0.85), (4,8)=>(0.18,0.37)
)

# --- STRUCTURES ---
mutable struct Produit
    id::Int
    type::Int
    arrivee_atelier::Float64
    arrivee_machine::Float64
    etape::Int
end

mutable struct Machine
    file_attente::Vector{Produit}
    en_service::Union{Produit,Nothing}
    occupee::Bool
    Machine() = new(Produit[], nothing, false)
end

mutable struct Employe
    id::Int
    qualifications::Vector{Int}
    temps_entree_salle::Float64
    est_occupe::Bool
    travail_cumule::Float64
    machine_assignee::Union{Int,Nothing}
    Employe(id, quals) = new(id, quals, 0.0, false, 0.0, nothing)
end

mutable struct Atelier
    sim::Simulation
    machines::Vector{Machine}
    employes::Vector{Employe}
    file_inactifs::Vector{Int}
    events_employes::Vector{Event}
    temps_sejour::Vector{Float64}
    temps_sejour_history::Vector{Tuple{Float64,Float64}}
    compteur_produits::Int
    fin_transitoire::Float64
    verbose::Bool
    record_history::Bool
    function Atelier(sim, Q, verbose, record_history)
        nb_machines = 8
        nb_emp = size(Q, 1)
        machines = [Machine() for _ in 1:nb_machines]
        employes = [Employe(i, findall(j -> Q[i,j]==1, 1:nb_machines)) for i in 1:nb_emp]
        events = [Event(sim) for _ in 1:nb_emp]
        new(sim, machines, employes, Int[], events, Float64[], Tuple{Float64,Float64}[], 0, 0.0, verbose, record_history)
    end
end

# --- FONCTION DE LOG ---
log_event(atelier::Atelier, msg::String) = atelier.verbose && println("[t=$(Printf.@sprintf("%.3f", now(atelier.sim)))] $msg")

# --- FONCTIONS AUXILIAIRES ---
function par_temps_entree(atelier::Atelier, id::Int)
    return atelier.employes[id].temps_entree_salle
end

function index_produit_doyen(file::Vector{Produit})
    isempty(file) && return nothing
    idx_min = 1
    val_min = file[1].arrivee_atelier
    for i in 2:length(file)
        val = file[i].arrivee_atelier
        if val < val_min
            val_min = val
            idx_min = i
        end
    end
    return idx_min
end

function entrer_salle_attente!(atelier::Atelier, emp_id::Int, temps::Float64)
    emp = atelier.employes[emp_id]
    emp.est_occupe = false
    emp.temps_entree_salle = temps
    push!(atelier.file_inactifs, emp_id)
    sort!(atelier.file_inactifs, by = id -> par_temps_entree(atelier, id))
    log_event(atelier, "Employé $emp_id entre en salle d'attente (file=$(atelier.file_inactifs))")
    return nothing
end

function sortir_salle_attente!(atelier::Atelier, emp_id::Int)
    filter!(e -> e != emp_id, atelier.file_inactifs)
    atelier.employes[emp_id].est_occupe = true
    log_event(atelier, "Employé $emp_id sort de la salle d'attente")
end

# --- MODIFICATION Q9 : CHOIX DE L'EMPLOYÉ (Politique) ---
function choisir_employe_pour_machine(atelier::Atelier, machine_id::Int, mode_choix::String)
    # On identifie les employés en salle d'attente capables de s'occuper de cette machine
    qualifies = [id for id in atelier.file_inactifs if machine_id in atelier.employes[id].qualifications]
    
    isempty(qualifies) && return nothing

    if mode_choix == "MOINS_CHARGE"
        # Règle Q8 : Sélection de l'employé avec le temps de travail cumulé minimal.
        # Minimiser (travail / t) à l'instant t revient strictement à minimiser le travail.
        meilleur_emp = qualifies[1]
        min_travail = atelier.employes[meilleur_emp].travail_cumule
        
        for id in qualifies[2:end]
            travail = atelier.employes[id].travail_cumule
            if travail < min_travail
                min_travail = travail
                meilleur_emp = id
            end
        end
        sortir_salle_attente!(atelier, meilleur_emp)
        return meilleur_emp
    else
        # Fallback classique (le premier qualifié arrivé en salle)
        emp_id = qualifies[1]
        sortir_salle_attente!(atelier, emp_id)
        return emp_id
    end
end

# --- MODIFICATION Q9 : ROUTAGE (Algorithme 3 cas) ---
function choisir_machine(atelier::Atelier, emp_id::Int, mode_algo::String, mode_choix::String)
    emp_ref = atelier.employes[emp_id]

    if mode_algo == "EQUILIBRAGE"
        machines_candidates = [m for m in emp_ref.qualifications if !isempty(atelier.machines[m].file_attente)]
        
        # 3ème cas : Aucune tâche possible
        isempty(machines_candidates) && return nothing

        machines_cas1 = Int[]
        machines_cas2_valides = Int[]

        # Analyse du statut exclusif ou partagé des machines candidates
        for m_id in machines_candidates
            # On cherche les autres employés en salle capables de traiter cette machine
            autres_dispos = [id for id in atelier.file_inactifs if m_id in atelier.employes[id].qualifications]

            if isempty(autres_dispos)
                # 1er cas : Notre employé est le seul à pouvoir s'en occuper maintenant
                push!(machines_cas1, m_id)
            else
                # 2ème cas : Machine partagée. Existe-t-il un collègue moins chargé ?
                collegue_moins_charge = any(id -> atelier.employes[id].travail_cumule < emp_ref.travail_cumule, autres_dispos)
                
                if !collegue_moins_charge
                    # L'employé actuel reste le plus légitime (le moins chargé ou égal)
                    push!(machines_cas2_valides, m_id)
                end
            end
        end

        # Résolution : Priorité absolue au Cas 1 (FIFO entre elles)
        if !isempty(machines_cas1)
            meilleure_machine = nothing
            arrivee_min = Inf
            for m_id in machines_cas1
                idx = index_produit_doyen(atelier.machines[m_id].file_attente)
                t = atelier.machines[m_id].file_attente[idx].arrivee_atelier
                if t < arrivee_min
                    arrivee_min = t
                    meilleure_machine = m_id
                end
            end
            return meilleure_machine
        end

        # Résolution : Cas 2 (FIFO entre les machines partagées validées)
        if !isempty(machines_cas2_valides)
            meilleure_machine = nothing
            arrivee_min = Inf
            for m_id in machines_cas2_valides
                idx = index_produit_doyen(atelier.machines[m_id].file_attente)
                t = atelier.machines[m_id].file_attente[idx].arrivee_atelier
                if t < arrivee_min
                    arrivee_min = t
                    meilleure_machine = m_id
                end
            end
            return meilleure_machine
        end

        # Si l'employé n'a que des machines du cas 2 et qu'il y a toujours un collègue
        # moins chargé disponible pour chacune, il cède sa place (retour au cas 3).
        return nothing
    end

    return nothing # Fallback sécurisé
end

function arriver_sur_machine!(atelier::Atelier, p::Produit, m_id::Int, temps::Float64, mode_choix::String)
    machine = atelier.machines[m_id]
    log_event(atelier, "Produit $(p.id) (type $(p.type)) arrive sur machine $m_id (étape $(p.etape))")
    
    if machine.en_service === nothing && !machine.occupee
        emp_id = choisir_employe_pour_machine(atelier, m_id, mode_choix)
        if emp_id !== nothing
            machine.occupee = true
            machine.en_service = p
            atelier.employes[emp_id].machine_assignee = m_id
            log_event(atelier, "Machine $m_id libre -> Employé $emp_id assigné")
            ev = atelier.events_employes[emp_id]
            if state(ev) == SimJulia.idle
                succeed(ev)
            end
        else
            push!(machine.file_attente, p)
            log_event(atelier, "Machine $m_id libre mais aucun employé disponible -> file d'attente")
        end
    else
        push!(machine.file_attente, p)
        log_event(atelier, "Machine $m_id occupée -> file d'attente")
    end
    return nothing
end

# --- PROCESSUS DE SIMULATION ---

@resumable function processus_employe(sim::Simulation, emp_id::Int, atelier::Atelier, mode_algo::String, mode_choix::String)
    emp = atelier.employes[emp_id]
    while true
        if !emp.est_occupe
            m_id = choisir_machine(atelier, emp_id, mode_algo, mode_choix)
            if m_id !== nothing
                machine = atelier.machines[m_id]
                machine.occupee = true
                emp.est_occupe = true
                
                idx_doyen = index_produit_doyen(machine.file_attente)
                p = machine.file_attente[idx_doyen]
                deleteat!(machine.file_attente, idx_doyen)
                
                machine.en_service = p
                emp.machine_assignee = m_id
                
                duree = rand(Uniform(TEMPS[(p.type, m_id)]...))
                log_event(atelier, "Employé $emp_id commence machine $m_id pour produit $(p.id)")
                
                @yield timeout(sim, duree)
                
                if now(sim) >= atelier.fin_transitoire
                    emp.travail_cumule += duree
                end
                
                machine.en_service = nothing
                machine.occupee = false
                emp.machine_assignee = nothing
                emp.est_occupe = false
                
                p.etape += 1
                if p.etape <= length(PARCOURS[p.type])
                    m_suiv = PARCOURS[p.type][p.etape]
                    p.arrivee_machine = now(sim)
                    arriver_sur_machine!(atelier, p, m_suiv, now(sim), mode_choix)
                else
                    t_fin = now(sim)
                    ts = t_fin - p.arrivee_atelier
                    if now(sim) >= atelier.fin_transitoire
                        push!(atelier.temps_sejour, ts)
                    end
                    if atelier.record_history
                        push!(atelier.temps_sejour_history, (t_fin, ts))
                    end
                end
                continue
            end
        end

        if !emp.est_occupe && emp.machine_assignee === nothing
            entrer_salle_attente!(atelier, emp_id, now(sim))
            atelier.events_employes[emp_id] = Event(sim)
            ev = atelier.events_employes[emp_id]
            @yield ev
            
            m_id = emp.machine_assignee
            emp.machine_assignee = nothing
            machine = atelier.machines[m_id]
            p = machine.en_service
            
            duree = rand(Uniform(TEMPS[(p.type, m_id)]...))
            @yield timeout(sim, duree)
            
            if now(sim) >= atelier.fin_transitoire
                emp.travail_cumule += duree
            end
            
            machine.en_service = nothing
            machine.occupee = false
            emp.est_occupe = false
            
            p.etape += 1
            if p.etape <= length(PARCOURS[p.type])
                m_suiv = PARCOURS[p.type][p.etape]
                p.arrivee_machine = now(sim)
                arriver_sur_machine!(atelier, p, m_suiv, now(sim), mode_choix)
            else
                t_fin = now(sim)
                ts = t_fin - p.arrivee_atelier
                if now(sim) >= atelier.fin_transitoire
                    push!(atelier.temps_sejour, ts)
                end
                if atelier.record_history
                    push!(atelier.temps_sejour_history, (t_fin, ts))
                end
            end
            continue
        end
    end
end

@resumable function generateur(sim::Simulation, type::Int, atelier::Atelier, mode_choix::String)
    while true
        @yield timeout(sim, rand(Exponential(1/λ[type])))
        atelier.compteur_produits += 1
        t_now = now(sim)
        p = Produit(atelier.compteur_produits, type, t_now, t_now, 1)
        premiere_machine = PARCOURS[type][1]
        arriver_sur_machine!(atelier, p, premiere_machine, t_now, mode_choix)
    end
end

@resumable function processus_reinitialisation(sim::Simulation, atelier::Atelier, duree_transient::Float64)
    @yield timeout(sim, duree_transient)
    log_event(atelier, "=== FIN PÉRIODE TRANSITOIRE - RÉINITIALISATION STATISTIQUES ===")
    for emp in atelier.employes
        emp.travail_cumule = 0.0
    end
end

# --- MOTEUR D'ÉVALUATION ---

function etude_performance(Q, label, mode_algo="EQUILIBRAGE", mode_choix="MOINS_CHARGE",
                           n_runs=20, duree_transient=10000.0, duree_permanent=1000.0;
                           verbose=false, plot_convergence=false)
    sejours_moyens = Float64[]
    nb_emp = size(Q, 1)
    occupations = [Float64[] for _ in 1:nb_emp]

    for r in 1:n_runs
        record_this_run = plot_convergence && (r == 1)
        sim = Simulation()
        atelier = Atelier(sim, Q, verbose, record_this_run)
        atelier.fin_transitoire = duree_transient

        @process processus_reinitialisation(sim, atelier, duree_transient)

        for i in 1:nb_emp
            @process processus_employe(sim, i, atelier, mode_algo, mode_choix)
        end
        for t in 1:4
            @process generateur(sim, t, atelier, mode_choix)
        end

        run(sim, duree_transient + duree_permanent)

        if !isempty(atelier.temps_sejour)
            push!(sejours_moyens, mean(atelier.temps_sejour))
        end
        for i in 1:nb_emp
            push!(occupations[i], atelier.employes[i].travail_cumule / duree_permanent * 100)
        end
    end

    m = mean(sejours_moyens)
    ic = 1.96 * std(sejours_moyens) / sqrt(n_runs)
    occ_moy = [mean(occupations[i]) for i in 1:nb_emp]

    println("-"^60)
    @printf("Instance %s | Algo: %-15s | Choix: %-15s\n", label, mode_algo, mode_choix)
    @printf(" > Temps de séjour moyen : %.4f ± %.4f\n", m, ic)
    @printf(" > Occupation employés   : %s\n", join([@sprintf("%.1f%%", x) for x in occ_moy], ", "))
end

# --- CONFIGURATION ET EXÉCUTION ---
Q1 = [1 1 0 0 0 0 0 0; 0 0 1 1 0 0 0 0; 0 0 0 0 1 1 0 0; 0 0 0 0 0 0 1 1]
Q2 = [1 0 1 0 0 1 0 0; 0 1 0 0 1 0 1 1; 0 1 0 1 1 0 0 1; 1 0 1 1 0 1 1 0]
Q3 = [1 1 0 0 0 0 0 0; 0 0 1 0 0 0 0 0; 0 0 0 1 0 1 0 0; 0 0 0 0 1 0 0 1; 0 0 1 0 1 1 0 0; 1 0 0 0 0 0 1 0]
Q4 = [1 1 1 0 0 0 0 0; 0 0 0 1 1 1 0 0; 1 0 1 0 0 1 1 1; 0 0 1 0 1 0 1 1; 0 1 0 0 0 1 1 0; 1 0 0 1 0 0 1 1]

Random.seed!(123)
println("="^60)
println("ALGO EQUILIBRAGE DES CHARGES (Q9)")
println("="^60)
etude_performance(Q1, "I1", "EQUILIBRAGE", "MOINS_CHARGE")
etude_performance(Q2, "I2", "EQUILIBRAGE", "MOINS_CHARGE")
etude_performance(Q3, "I3", "EQUILIBRAGE", "MOINS_CHARGE")
etude_performance(Q4, "I4", "EQUILIBRAGE", "MOINS_CHARGE")