import streamlit as st 

from main import (
    prepare_sleep_df, 
    calculer_stats_globales, 
    calculer_stats_par_periode, 
    calculer_heures_medianes, 
    creer_rapport, 
    visualiser_distribution_duree, 
    visualiser_evolution_duree, 
    visualiser_coucher_vs_duree,
    ecrire_rapport_txt,
    ecrire_rapport_pdf,
    horodater,
    traduire_date_en_francais,
    traduire_colonne_date_en_francais
)


def main() : 
    st.title("DataPulse Sleep Analysis")
    st.sidebar.header("Paramètres")
    
    
    uploaded_file  =st.sidebar.file_uploader("Fichier csv de sommeil", type = ["csv"])
    
    format_rapport = st.sidebar.radio ("Format de rapport " , ["Texte","pdf", "les 2"])
    
    afficher_viz =st.sidebar.checkbox("Generer des visualisations" , value = True) 
    
        
    if st.sidebar.button("Analyser") : 
        
        if uploaded_file is not None : 
            df = prepare_sleep_df(uploaded_file)
            if df is None: 
                st.error("Eerrur lors du traitement du fichier!")
                st.stop()
            globlales = calculer_stats_globales(df)
            periodiques  = calculer_stats_par_periode(df)
            heures_med = calculer_heures_medianes(df)
            stats = globlales+periodiques + heures_med
            
            rapport = creer_rapport (
                stats,
                df['date'].min().strftime('%A %d %B %Y'),
                df['date'].max().strftime('%A %d %B %Y'))
            
            # ✅ Résumé tabulaire interactif
            df_display = df.copy()
            df_display['duree_h'] = "~" + (df_display['duree'] / 60).round(1).astype(str) + " h"
            df_display['date'] = traduire_colonne_date_en_francais(df_display['date'])

            #df_display['date'] = df_display['date'].dt.strftime('%A %d %B %Y').apply(traduire_date_en_francais)
            df_display['coucher'] = df_display['coucher'].dt.strftime('%H:%M:%S')
            df_display['lever'] = df_display['lever'].dt.strftime('%H:%M:%S')
            
            colonnes_affichage = ["date", "coucher", "lever", "duree_h", "qualite"]
            st.subheader("📊 Nuits analysées")
            st.dataframe(df_display[colonnes_affichage])

            if afficher_viz:
                visualiser_distribution_duree(df)
                visualiser_evolution_duree(df)
                visualiser_coucher_vs_duree(df)
                col1, col2 ,col3= st.columns(3)
                with col1:
                   st.image("data/out/distribution_duree_sommeil.png", caption="Distribution", use_container_width=True)
                with col2:
                   st.image("data/out/evolution_duree_sommeil.png", caption="Évolution", use_container_width=True)
                with col3:
                   st.image("data/out/sommeil_heure_coucher.png", caption="Heure de coucher", use_container_width=True)

                st.success("✅ Graphiques générés")
            
            if format_rapport in ["Texte", "les 2"]:
                ecrire_rapport_txt(horodater("data/out/DataPulse_Sleep_Report.txt"), rapport)
            if format_rapport in ["pdf", "les 2"]:
                ecrire_rapport_pdf(horodater("data/out/DataPulse_Sleep_Report.pdf"), rapport)
            st.subheader("📄 Rapport Synthétique")
            st.text_area("Contenu du rapport", rapport, height=300)
            
            
        else : 
            st.error ("Veuillez sélectionner un fichier csv.")
            
            
if __name__ == "__main__" : 
    main()