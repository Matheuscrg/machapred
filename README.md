# MaChaPred(English version)

A Deep learning app for predicting bioactive compounds against Plasmodium sp. and Trypanosoma cruzi

The online version of this app  is available at http://machapred.app/

To run locally, you have to install Chemprop version 2.2.2 and run this command replacing with your data:

!chemprop predict --test-path path/to/your/file.csv --model-path path/to/the/model.pt --preds-path path/to/where/you/want/your/results.csv --smiles-columns NAME_OF_YOUR_SMILES_COLUMN

To run the online version you first need to install Streamlit and its dependencies, and Chemprop version 2.2.2.
After that, you must create a folder named models, and then put the models.pt inside this folder.

After that, you can run the following command to make it run:

!streamlit run app.py --server.address 0.0.0.0 --server.port 80 #Chage for the IP and port of your choosing.

If you want to run it indefinitely, an easy solution is to use the nohup package.


# MaChaPred versão em português:

Um aplicativo de Deep Learning para prever compostos bioativos contra Plasmodium sp. e Trypanosoma cruzi.

A versão online deste aplicativo está disponível em http://machapred.app/

Para rodar localmente, você precisa instalar o Chemprop versão 2.2.2 e executar este comando, substituindo pelos seus dados:

!chemprop predict --test-path caminho/para/seu/arquivo.csv --model-path caminho/para/o/modelo.pt --preds-path caminho/para/onde/voce/quer/seus/resultados.csv --smiles-columns NOME_DA_SUA_COLUNA_SMILES


Para rodar a versão online, primeiro você precisa instalar o Streamlit e suas dependências, além do Chemprop versão 2.2.2.
Depois disso, você deve criar uma pasta chamada models e colocar os modelos.pt dentro desta pasta.

Em seguida, você pode executar o seguinte comando para fazer o servidor funcionar:

!streamlit run app.py --server.address 0.0.0.0 --server.port 80 # Mude para o IP e a porta de sua escolha.

Se você quiser que ele rode indefinidamente, uma solução fácil é usar o pacote nohup.



